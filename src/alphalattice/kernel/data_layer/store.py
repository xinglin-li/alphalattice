"""StorePort: the one door to the catalog and its files, the write order, and its one repair.

The data root's files, by who names them:
    staging/<admission_id>/  one admission's Parquet before its catalog commit
    staging/<admission_id>.lock  held by its producer from before the folder exists until its
                             commit returns; staging whose lock no one holds belongs to a write
                             that will never finish, and start collects it
    data/                    DuckLake's files, named by its catalog; any other file is collected
    blobs/<artifact_id>      an artifact's body, kept while a live_ref holds it
    blobs/.<artifact_id>.<pid>.staging  an array's body before its profile commits (`arrays`)

The write order (catalog.sql's header). `stage` mints the admission's id and writes its rows'
Parquet under staging. It needs no log, lock or catalog, so it runs beside the writer, in the
process that fetched the rows, and hands the writer only a commit request (`Staged`): in the
Host's process the rows' Python and Arrow work would hold the interpreter while an append waits.
`StorePort.commit` is the writer: one catalog snapshot conforms the rows to their table's types,
names each new version by an id the engine mints, inserts the rows, writes the supersession of
every world version they replace and carries in its commit info the whole envelope of the
`admission.committed` it owes, in a versioned list (`owed_events`): one envelope for a lone commit,
each admission's for a block of them (`StorePort.admitting`), which share one snapshot while each
keeps its own id, provider and lag rule. Then `repair` appends each event, durably, and the staging
folders are deleted. `start` runs the same `repair`
when the Host opens, so the normal path and crash recovery are one path, and collects what no
snapshot or write owns. Every file leaves through `delete`: one another process holds open waits
in deletion_queue and is retried at each start, and nothing a live_ref holds is ever deleted.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Final, NamedTuple

import alphalattice.kernel.data_layer.governance as governance
from alphalattice.control.workspace_runtime.storage.capacity import (
    FREE_FLOOR_BYTES,
    free_floor_shortfall,
)
from alphalattice.kernel import storage
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import ENVELOPE, canonical
from alphalattice.kernel.data_layer.log import HostLog, drop, envelope, hold, uuid7
from alphalattice.kernel.data_layer.root import (
    attach_catalog,
    bound_reads,
    catalog,
    catalog_matches,
    configure_read_cache,
    lake_thread_config,
    load_extensions,
    schema_text,
    sql_path,
    statements,
    view_statements,
    world_table_ids,
)
from alphalattice.kernel.data_layer.telemetry import usage

_GROUPING = threading.Lock()
_grouping: list[threading.Thread] = []


def preload_grouping() -> None:
    """Load Arrow's grouping engine, which an admission groups its rows with, on a thread of its
    own, once a process: the Host calls it once it serves, so its first admission does not pay
    the load (about a second, pandas with it) and its start does not either."""
    with _GROUPING:
        if _grouping or find_spec("pyarrow") is None:
            return
        thread = threading.Thread(target=_load_grouping, name="arrow-grouping", daemon=True)
        _grouping.append(thread)
    thread.start()


def _load_grouping() -> None:
    import pyarrow.acero  # noqa: F401  # lazy: the Host preloads it once it serves


KINDS: Final = {"staging": "staging", "blobs": "blob", "data": "file"}
"""The deletion queue's kind for each top folder a deleted file may live in."""


def _admission_rules() -> dict[str, list[dict[str, Any]]]:
    """Load each family's table registrations in the existing admission order."""
    groups: dict[str, list[dict[str, Any]]] = {"world": [], "results": [], "derived": []}
    schema = resources.files(__package__).joinpath("schema")
    for path in sorted(schema.iterdir(), key=lambda item: item.name):
        if path.name.startswith("admission-") and path.name.endswith(".json"):
            try:
                for group, entries in json.loads(path.read_text(encoding="utf-8")).items():
                    groups[group].extend(entries)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                raise DataError(
                    f"cannot read StorePort admission registrations in {path.name}",
                    code="data_layer.schema_mismatch",
                ) from error
    for group, entries in groups.items():
        entries.sort(key=lambda entry: (entry["order"], entry["table"]))
        if len({entry["table"] for entry in entries}) != len(entries):
            raise DataError(
                f"duplicate StorePort admission registration in {group}",
                code="data_layer.schema_mismatch",
            )
    return groups


_ADMISSION_RULES: Final = _admission_rules()

WORLD_KEYS: Final = {rule["table"]: tuple(rule["key"]) for rule in _ADMISSION_RULES["world"]}
"""Each world table catalog.sql defines, with its key. An observed row is known from its
admission's moment: known_at and known_asof are both the log's clock."""
ADMITTED: Final = (
    "version_id",
    "known_at",
    "known_asof",
    "lag_rule",
    "lag_rule_version",
    "admission_id",
)
"""The columns StorePort writes on each world row at its admission; no writer sends them. A
legacy row names the lag rule and version its known_asof was derived by."""
SOURCE: Final = ("source_observed_at", "source_payload_digest")
"""The world columns that record how a source wrote a value, never the fact. A world table's
content is its every other catalog column (its key, its values and `retracted`): a staged row is
a new version only when its content differs from its key's newest version's."""
LINEAGES: Final = {
    "provider_id": "provider",
    "scheme_id": "sector_scheme",
    "universe_id": "universe",
}
"""The world columns that name a lineage, with its family: a row names a registered one."""
RESULTS_KEYS: Final = {rule["table"]: tuple(rule["key"]) for rule in _ADMISSION_RULES["results"]}
"""The results tables StorePort admits, with their keys. A results row is an artifact's body,
written once: a key already in the catalog is refused. Its lineage is its artifact_id, not an
outside provider: its admission names no provider and lists the artifacts it holds."""
DERIVED_KEYS: Final = {rule["table"]: tuple(rule["key"]) for rule in _ADMISSION_RULES["derived"]}
"""The derived caches StorePort admits, with their keys: a feature set's table by its prefix
(feature_table). A cache is rebuildable and no row references it, so its key may hold a digest;
its rows are written only by `StorePort.derive`, and the log's cache_entry says which are valid."""
DERIVED_PARTS: Final = {rule["table"]: rule["part"] for rule in _ADMISSION_RULES["derived"]}
"""Each derived table's partition of a row, as cache_entry's `part` names it: a year, a month and
listing (a feature set's: a build's batch writes whole listings, and a day's append moves one
month of each), or an as-of date."""
CHANGES: Final = "__ducklake_metadata_lake.main.ducklake_snapshot_changes"
"""Each snapshot's commit message and info, read in the catalog database DuckLake attaches for
`lake` under its documented name (METADATA_CATALOG's default, `__ducklake_metadata_<name>`)."""
KEEP_SNAPSHOTS: Final = timedelta(days=7)
"""How long the catalog keeps a snapshot: the plan's snapshot rule (0.2.md, 0.2.8 decision 2)."""
SETTLE_FILES: Final = timedelta(days=3)
"""How long a file no kept snapshot names waits before it is deleted, so a read that listed it
before its snapshot expired still finds it: DuckLake's own guidance, a few days."""
IDLE_MAINTENANCE_AGE: Final = timedelta(days=1)
IDLE_MAINTENANCE_WRITES: Final = 16
"""Earlier of one wall day or sixteen catalog snapshots, checked at the Host's next idle turn."""
IDLE_MAINTENANCE_THREADS: Final = 4
DERIVED_REWRITE: Final = 0.5
"""The share of a derived file's rows that must be dead before maintenance rewrites it: a
collected partition is the derived schema's normal churn, so its dead bytes never outgrow the live
ones. World and results keep DuckLake's own 0.95, since their rows are rarely deleted."""
LEGACY_VALID: Final = {
    "listing": ("last_session", "first_session"),
    "provider_symbol": ("valid_to", "valid_from"),
    "universe_member": ("valid_to", "valid_from"),
    "sector": ("valid_to", "valid_from"),
    "raw_bar": ("session_date",),
    "corporate_action": ("effective_date",),
    "evidence_doc": ("available_at",),
}
"""The world tables a legacy batch may fill, each with the date a row's lag rule counts from: the
first of these the row has, so an interval's closing version is known only from its end."""
UNMARKETED: Final = frozenset({"evidence_doc"})
"""The legacy tables whose rows name no market: a row is known from its LEGACY_VALID time itself
plus its lag, never a session's close, and its publication time never dates it. A filing's
availability is its acceptance, later than its filing date's midnight."""
OBSERVED: Final = SOURCE[0]
TIMED: Final = frozenset({"raw_bar", "corporate_action"})
"""The world tables whose rows carry their source's own last write of the value (OBSERVED): a held
key is superseded only by a row its source wrote after the newest version's value was last
written. The time is SOURCE, never content."""
PUBLISHED: Final = "published_batch"
"""The name a published batch is registered under on the catalog session while it is judged."""
KNOWN: Final = "_known_us"
"""The column of a published batch (`StorePort.publish`) carrying each row's observed time, in
integer microseconds: when its writer observed it, which dates the row in place of the commit's
clock. Never written to the catalog."""
REWRITTEN: Final = "rewritten"
"""A row of a TIMED table whose source wrote its key's newest value again, later: world.rewritten
records the write against that version, which adds no version and moves its ordering time."""
DROPPED: Final = ("older", "equal_time")
"""Why a changed row of a held key was not admitted: its source wrote it before the newest version,
or at the same time with another value, which keeps the held version."""
PARTITION: Final = {"raw_bar": "session_date"}
"""The date column whose year partitions a table, as catalog.sql sets it; any other table is one
part, `all`. An admission's event names the knowledge times it brought to each part."""
METADATA: Final = "__ducklake_metadata_lake"
"""The DuckLake catalog's own tables, as the Host's session names them."""


class Staged(NamedTuple):
    """One admission's rows, written under staging, waiting for its commit."""

    admission_id: str
    table: str
    provider_id: str | None
    """A world admission's provider lineage, as the log registers it; None for a results one."""
    row_count: int
    lag: tuple[str, int] | None
    """A legacy batch's lag rule, by name and version; None for an observed one."""
    artifacts: tuple[str, ...] = ()
    """A results admission's artifact ids, its rows' lineage; empty for a world one."""


class StagedSet(NamedTuple):
    """Several results tables staged under one admission and one producer lock."""

    admission_id: str
    tables: tuple[Staged, ...]


class Derivation(NamedTuple):
    """One derived table's rows for `StorePort.derive_set`, each field as `derive` means it."""

    table: str
    rows: Any
    recipe_key: str
    parts: Any
    compute_ms: int


class Committed(NamedTuple):
    """What one commit did: its snapshot, None when it wrote nothing or joined a block of
    admissions (`StorePort.admitting`), whose end commits it; the rows it admitted; the
    changed rows of held keys it dropped, by DROPPED's reasons; and the later writes of held values
    it recorded (REWRITTEN)."""

    snapshot: int | None
    rows: int
    dropped: dict[str, int]
    rewritten: int


OWED_EVENTS_VERSION: Final = 2
"""The version of an admission commit's info: the list of events it owes, one per admission. An
info written before it is one envelope, which reads as a list of one (`owed_events`)."""


def owed_events(info: str) -> list[dict[str, Any]]:
    """The `admission.committed` envelopes an admission snapshot's commit info carries, in their
    admissions' order. Refused by name: a version this Host does not read."""
    carried = json.loads(info)
    if "event_id" in carried:  # one envelope, as a commit before OWED_EVENTS_VERSION wrote it
        return [carried]
    if carried.get("version") != OWED_EVENTS_VERSION:
        raise DataError(
            f"an admission snapshot's commit info has version {carried.get('version')}, which"
            " this Host does not read; open the data root with the release that wrote it",
            code="data_layer.owed_events_version",
        )
    return list(carried["events"])


def expirable(
    snapshots: list[tuple[int, datetime, str | None]], held: set[str], now: datetime
) -> list[int]:
    """The snapshots maintenance may expire, of (id, time, an event its admissions owe or None),
    a snapshot listed once per event: those older than KEEP_SNAPSHOTS, never the newest, and none
    from the oldest snapshot owing an event the log does not hold (`held`) on, since the next
    start's repair reads its commit."""
    owed = min((s for s, _, event in snapshots if event and event not in held), default=None)
    newest = max((s for s, _, _ in snapshots), default=None)
    return sorted(
        {
            s
            for s, at, _ in snapshots
            if at < now - KEEP_SNAPSHOTS and s != newest and (owed is None or s < owed)
        }
    )


class _Block:
    """The admissions one `StorePort.admitting` block has made: the events its snapshot owes, the
    staging folders its commit settles, the records naming its rows that its commit publishes
    (`StorePort.after_commit`), and whether an admission's writes are under way, which a failure
    then rolls back with the whole block."""

    def __init__(self) -> None:
        self.owed: list[dict[str, Any]] = []
        self.staging: list[str] = []
        self.published: list[Callable[[], None]] = []
        self.writing = False


class StorePort:
    """The Host's one writer of the catalog; open it with `StorePort.open`."""

    def __init__(self, root: Path, log: HostLog, lake: Any):
        """Hold the data root, its log and the Host's DuckDB session; `open` makes one."""
        self.root, self.log, self.lake = root, log, lake
        with log.reading() as con:
            self.table_ids = world_table_ids(con)
        self._writer = threading.RLock()
        self.started: dict[str, list[str]] = {}
        self._repaired = 0  # every admission snapshot up to this one has its event in the log
        # What this session read of the catalog's world tables, read once: a table's columns
        # (`_declared`). This session is the catalog's one writer (a second Host is refused), so
        # they change only by its own work: its DDL clears the columns (`_feature_set`).
        self._columns: dict[str, dict[str, str]] = {}
        self._block: _Block | None = None  # the open `admitting` block, on the writer's thread
        self._maintained: tuple[int, datetime] | None = None

    @classmethod
    def open(cls, root: Path, log: HostLog, *, owner: Any = None) -> StorePort:
        """Attach the catalog and the reading surface's world macros, then run `start`. The log
        is never attached: this process writes it through Python's sqlite3, and a second SQLite
        library closing its descriptor would release the writer's locks on the file."""
        lake = (
            owner.cursor()
            if owner is not None
            else storage.connect(  # through the driver: DuckDB, the optional data extra
                storage.MEMORY,
                config={
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                    **lake_thread_config(),
                },
            )
        )
        configure_read_cache(lake)
        if owner is None:
            load_extensions(lake)
            bound_reads(lake)
        if not catalog(root, owner=lake, keep_attached=True):
            attach_catalog(lake, root)
        lake.execute("SET TimeZone = 'UTC'").complete()
        catalog_matches(lake)  # before the views, which name its columns
        if not lake.execute(
            "SELECT 1 FROM duckdb_tables() WHERE database_name = 'lake'"
            " AND schema_name = 'results' AND table_name = 'validation_receipt'"
        ).fetchone():  # the one additive 0.2.8 catalog upgrade
            ddl = next(
                sql
                for sql in statements(schema_text("catalog.sql"))
                if sql.partition("(")[0].strip().endswith("results.validation_receipt")
            )
            lake.execute(
                ddl.replace("results.validation_receipt", "lake.results.validation_receipt", 1)
            ).complete()
        port = cls(root, log, lake)
        for sql in view_statements(port.table_ids):
            if "log." not in sql:
                lake.execute(sql).complete()
        port.started = start(root, log, lake)
        for sql in statements(schema_text("telemetry.sql")):
            lake.execute(sql).complete()
        lake.execute(
            "CREATE VIEW agent_telemetry AS SELECT *,"
            " CASE WHEN vendor = 'codex' THEN CASE WHEN input_tokens >="
            " coalesce(cache_read_tokens, 0) + coalesce(cache_write_tokens, 0) THEN input_tokens"
            " - coalesce(cache_read_tokens, 0) - coalesce(cache_write_tokens, 0) END"
            " ELSE input_tokens END AS uncached_input_tokens,"
            " conversation_id IS NULL AND session_id IS NULL AS unplaced"
            ", CASE WHEN name IN ('claude_code.api_request', 'codex.api_request')"
            " THEN 'gen_ai.client.inference' WHEN name = 'claude_code.subagent_completed'"
            " THEN 'gen_ai.agent.completed' ELSE name END AS event_name"
            " FROM lake.telemetry.event"
        ).complete()
        return port

    def telemetry_batch(self, rows: list[dict[str, Any]]) -> int:
        """Commit one already sanitized native export under the catalog's existing writer.

        The whole HTTP batch is one snapshot. Only the decoder's fixed columns may enter;
        repeated delivery keeps the original rows. No raw payload is staged on disk.
        """
        if not rows:
            return 0
        import pyarrow as pa  # lazy: the Host's existing data extra

        with self._writer, self.lake.cursor() as lake:
            received_us = self.log.clock()
            batch = pa.Table.from_pylist(
                [
                    row | {"received_us": received_us, "record_index": i}
                    for i, row in enumerate(rows)
                ]
            )
            lake.register("agent_batch", batch)
            try:
                lake.execute("BEGIN TRANSACTION").complete()
                ((before,),) = lake.execute("SELECT count(*) FROM lake.telemetry.event").fetchall()
                lake.execute(
                    "INSERT INTO lake.telemetry.event BY NAME"
                    " SELECT DISTINCT b.* FROM agent_batch b WHERE NOT EXISTS"
                    " (SELECT 1 FROM lake.telemetry.event e WHERE e.event_id = b.event_id)"
                ).complete()
                ((after,),) = lake.execute("SELECT count(*) FROM lake.telemetry.event").fetchall()
                lake.execute("COMMIT").complete()
                return int(after - before)
            except BaseException:
                lake.execute("ROLLBACK").complete()
                raise
            finally:
                lake.unregister("agent_batch")

    def agent_telemetry(
        self, host: str, session: str, *, workspace_id: str
    ) -> list[dict[str, Any]]:
        """Read this workspace's exact vendor and conversation/session, in event order."""
        with self._writer, self.lake.cursor() as lake:
            result = lake.execute(
                "SELECT * FROM agent_telemetry WHERE workspace_id = ? AND vendor = ?"
                " AND (conversation_id = ? OR session_id = ? OR parent_agent_id = ?)"
                " ORDER BY received_us, record_index",
                [workspace_id, host, session, session, session],
            )
            columns = [column[0] for column in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def unplaced_telemetry(self, *, workspace_id: str) -> int:
        """Count only this workspace's observations without a conversation or session."""
        with self._writer, self.lake.cursor() as lake:
            ((count,),) = lake.execute(
                "SELECT count(*) FROM agent_telemetry WHERE workspace_id = ? AND unplaced",
                [workspace_id],
            ).fetchall()
            return int(count)

    def trial_agent_usage(self, trial: str, *, workspace_id: str) -> dict[str, Any]:
        """The observed native sums a trial closes with, with no invented zero on absence."""
        with self._writer, self.lake.cursor() as lake:
            result = lake.execute(
                "SELECT * FROM agent_telemetry WHERE workspace_id = ? AND trial_id = ?"
                " ORDER BY received_us, record_index",
                [workspace_id, trial],
            )
            columns = [column[0] for column in result.description]
            rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        observed = usage(rows)
        if not observed["models"] or observed["incomplete"]:
            return {"status": "UNMEASURED", "incomplete": observed["incomplete"]}
        return {
            "status": "MEASURED",
            "model_seconds": observed["model_seconds"],
            **{
                field: sum(row[field] for row in observed["models"])
                for field in (
                    "responses",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            },
        }

    @contextmanager
    def admitting(self) -> Iterator[None]:
        """Commit every admission the block makes in one catalog snapshot. Each keeps its own
        admission id, provider and lag rule, and is judged against the world the block's earlier
        admissions wrote; the snapshot's commit info carries their events in order, which
        `repair` appends. A commit inside the block returns no snapshot. The writer is held for
        the whole block, which a block inside it joins.

        An exception outside an admission's writes, a refusal by name included, commits the
        admissions made before it, as committing each alone would have, then is raised. One
        inside an admission's writes, which only the engine raises, or an interruption rolls the
        block back whole: none of its admissions is recorded, and its producer makes them again.
        """
        with self._writer:
            if self._block is not None:
                yield
                return
            block = _Block()
            self.lake.execute("BEGIN").complete()
            self._block = block  # published once its transaction is open
            try:
                yield
            except Exception:
                self._block = None
                if block.writing:
                    self._rolled_back()
                else:
                    self._close_block(block)
                raise
            except BaseException:
                self._block = None
                self._rolled_back()
                raise
            self._block = None
            self._close_block(block)

    def _close_block(self, block: _Block) -> None:
        """Commit a block's admissions as one snapshot carrying their events, its all-dropped
        judgments made after its first write among them; then append them, delete its staging
        and publish, in order, the records naming its rows."""
        run = self.lake.execute
        try:
            if block.owed:
                info = {"version": OWED_EVENTS_VERSION, "events": block.owed}
                run(
                    "CALL lake.set_commit_message('host', 'admission', extra_info => ?)",
                    (canonical(info).decode(),),
                ).complete()
            run("COMMIT").complete()
        except BaseException:
            self._rolled_back()
            raise
        repair(self.log, self.lake, self._repaired)
        for admission in block.staging:
            delete(self.root, self.log, f"staging/{admission}")
        if block.owed:
            found = self.log.read(
                "SELECT catalog_snapshot FROM admission WHERE admission_id = ?",
                (block.owed[-1]["subject_id"],),
            )
            self._repaired = int(found[0][0])  # the repair closed every gap up to this commit
        for publish in block.published:
            publish()

    def after_commit(self, publish: Callable[[], None]) -> None:
        """Run `publish`, which records what names rows admitted so far, once they are committed:
        now, or when the block admitting them commits, in the order given. A block rolled back
        publishes none; its producer admits the rows again, and publishes then."""
        with self._writer:
            if self._block is None:
                publish()
            else:
                self._block.published.append(publish)

    def _rolled_back(self) -> None:
        """Roll back an open block."""
        self.lake.execute("ROLLBACK").complete()

    def market_id(self, mic: str, provider_id: str) -> str:
        """Resolve a recorded MIC binding, or register a new market under the Host's writer.

        Only a current version proves a binding. A historical name alone or several current
        bindings cannot identify a source's market; the source must supply a verified binding.
        No issue date is invented for an initial MIC whose start was not recorded.
        """
        import pyarrow as pa  # lazy: the optional data extra, only for market admission

        with self._writer:
            held = self.lake.execute(
                f"SELECT DISTINCT CAST(market_id AS VARCHAR) FROM {_current('market')}"
                " WHERE mic = ?",
                [mic],
            ).fetchall()
            if len(held) > 1:
                raise DataError(
                    f"MIC {mic} names several recorded markets; bind the source to its market id",
                    code="data_layer.market_mic_ambiguous",
                )
            if held:
                return str(held[0][0])
            if self.lake.execute(
                "SELECT 1 FROM lake.world.market WHERE mic = ? LIMIT 1", [mic]
            ).fetchone():
                raise DataError(
                    f"MIC {mic} has no current binding; register its verified market binding",
                    code="data_layer.market_mic_ambiguous",
                )
            market_id = uuid7()
            with stage(
                self.root,
                "market",
                provider_id,
                pa.table({"market_id": [market_id], "mic": [mic]}),
            ) as staged:
                self.commit(staged)
            return market_id

    def commit(self, staged: Staged | StagedSet) -> Committed:
        """Commit one staged admission as one catalog snapshot, then its owed event. A world row
        already its key's newest version is not admitted again, so a repeated batch adds nothing
        and takes no snapshot. The event is durable, so it is on disk when this returns. Refused
        by name: a lag rule the log does not hold, legacy rows with no session to count from,
        results keys already admitted, and a provider, scheme or universe no lineage in the log
        registers."""
        with self._writer:
            tables = staged.tables if isinstance(staged, StagedSet) else (staged,)
            for table in tables:
                self._registered(table, _staged_rows(self.root, table))
            if isinstance(staged, StagedSet) or staged.table in RESULTS_KEYS:
                count = self._catalog_results(staged, tables, self.log.clock())
                dropped, rewritten = {}, 0
            else:
                lag = None if staged.lag is None else self._lag(*staged.lag)
                source = _staged_rows(self.root, staged)
                count, dropped, rewritten = self._catalog_commit(
                    staged, self.log.clock(), lag, source
                )
            if self._block is not None:  # the block's one commit repairs and collects
                self._block.staging.append(staged.admission_id)
                return Committed(None, count, dropped, rewritten)
            return self._committed(staged, count, dropped, rewritten, staging=True)

    def publish(
        self, table: str, provider_id: str, rows: Any, lag: tuple[str, int] | None = None
    ) -> Committed:
        """Commit a world batch held in memory, an Arrow table, as `commit` commits a staged one,
        with no staging file: its rows are judged once, against world as an open block
        (`admitting`) has written it, and written in that block's transaction, or in one snapshot
        of their own. A batch carrying KNOWN dates each row's change by its observed time, never by
        this commit's clock. Refused by name as `admissible` and `commit` refuse."""
        with self._writer:
            data = admissible(table, provider_id, rows, legacy=lag is not None)
            staged = Staged(uuid7(), table, provider_id, data.num_rows, lag)
            self.lake.register(PUBLISHED, data)
            try:
                self._registered(staged, PUBLISHED)
                rule = None if lag is None else self._lag(*lag)
                count, dropped, rewritten = self._catalog_commit(
                    staged, self.log.clock(), rule, PUBLISHED
                )
            finally:
                self.lake.unregister(PUBLISHED)
            if self._block is not None:  # the block's one commit repairs
                return Committed(None, count, dropped, rewritten)
            return self._committed(staged, count, dropped, rewritten)

    def _committed(
        self,
        staged: Staged | StagedSet,
        count: int,
        dropped: dict[str, int],
        rewritten: int,
        *,
        staging: bool = False,
    ) -> Committed:
        """What a commit outside a block did, once its owed event is appended and its `staging`
        folder, if it had one, deleted."""
        repair(self.log, self.lake, self._repaired)
        if staging:
            delete(self.root, self.log, f"staging/{staged.admission_id}")
        if not count and not rewritten:
            return Committed(None, 0, dropped, 0)
        found = self.log.read(
            "SELECT catalog_snapshot FROM admission WHERE admission_id = ?",
            (staged.admission_id,),
        )
        self._repaired = int(found[0][0])  # the repair closed every gap up to this commit
        return Committed(self._repaired, count, dropped, rewritten)

    def derive(
        self,
        table: str,
        rows: Any,
        *,
        recipe_key: str,
        parts: Any,
        compute_ms: int,
        rooted_by: str | None = None,
    ) -> int:
        """Write derived partitions of `table`, computed by `recipe_key`, as one catalog snapshot,
        and record each valid in the log's cache_entry; the rows written. `parts` holds, as Arrow
        columns (part, input_digest), each partition the rows fill and the digest of what it was
        computed from: one partition, or a build's batch of them. A feature set's table is made,
        partitioned by year(session_date), and widened by a factor column, from the rows. A
        partition is written whole, so one already held under its digest is not written again:
        a repeated partition adds nothing, and workspaces whose partition has one digest share
        its rows. Refused by name: a table StorePort does not
        derive, and rows of another recipe, or of a partition or inputs `parts` does not name,
        which a cache entry would otherwise certify. `rooted_by` names the workspace whose reads
        take the partitions: they are rooted, and the root made durable, while the catalog's
        writer is still held, so no collection can see them unrooted."""
        derivation = Derivation(table, rows, recipe_key, parts, compute_ms)
        return self.derive_set((derivation,), rooted_by=rooted_by)[0]

    def derive_set(
        self, derivations: Sequence[Derivation], *, rooted_by: str | None = None
    ) -> list[int]:
        """`derive` for several derived tables in one catalog snapshot: each derivation checked
        and written as `derive` writes it, all of them in one transaction, their cache entries
        recorded and `rooted_by`'s roots made durable together; the rows written, in order. A
        commit costs a fixed part and a part per table and column written, so
        a stage writing several tables pays the fixed part once. Refused by name as `derive` is,
        before anything is written."""
        for derivation in derivations:
            if _family(derivation.table) not in DERIVED_KEYS:
                raise DataError(
                    f"{derivation.table} is not a derived table StorePort writes; it writes"
                    f" {', '.join(DERIVED_KEYS)}",
                    code="data_layer.derived_table_not_admitted",
                )
        run = self.lake.execute
        names = [(f"deriving_rows_{at}", f"deriving_parts_{at}") for at in range(len(derivations))]
        counts: list[int] = []
        filled: list[list[tuple[Any, ...]]] = []
        with self._writer:
            for (rows, parts), derivation in zip(names, derivations, strict=True):
                self.lake.register(rows, derivation.rows)
                self.lake.register(parts, derivation.parts)
            try:
                for (rows, parts), derivation in zip(names, derivations, strict=True):
                    self._named(derivation, rows, parts)
                for (rows, _), derivation in zip(names, derivations, strict=True):
                    if derivation.table.startswith("feature_set_"):
                        self._feature_set(derivation.table, rows)
                run("BEGIN").complete()
                try:
                    infos = []
                    for (rows, parts), derivation in zip(names, derivations, strict=True):
                        counts.append(self._derived(derivation, rows, parts))
                        if counts[-1]:
                            info = {"table": derivation.table, "recipe_key": derivation.recipe_key}
                            infos.append(
                                info | {"rows": counts[-1], "parts": len(derivation.parts)}
                            )
                    if infos:  # one table's commit says what it always said
                        info = infos[0] if len(infos) == 1 else {"tables": infos}
                        run(
                            "CALL lake.set_commit_message('host', 'derived', extra_info => ?)",
                            (canonical(info).decode(),),
                        ).complete()
                        run("COMMIT").complete()
                    else:
                        run("ROLLBACK").complete()
                except BaseException:
                    run("ROLLBACK").complete()
                    raise
                for (rows, _), derivation in zip(names, derivations, strict=True):
                    part = DERIVED_PARTS[_family(derivation.table)]
                    filled.append(
                        run(
                            f"SELECT {part}, input_digest, count(*) FROM {rows} GROUP BY ALL"
                        ).fetchall()
                    )
            finally:
                for rows, parts in names:
                    self.lake.unregister(rows)
                    self.lake.unregister(parts)
            for derivation, held in zip(derivations, filled, strict=True):
                share = int(getattr(derivation.rows, "nbytes", 0)) / max(len(derivation.rows), 1)
                self.log.cached(
                    derivation.table,
                    derivation.recipe_key,
                    [(name, digest, n, round(share * n)) for name, digest, n in held],
                    compute_ms=derivation.compute_ms,
                )
            if rooted_by is not None:
                for derivation, held in zip(derivations, filled, strict=True):
                    roots = [(derivation.recipe_key, name, digest) for name, digest, _ in held]
                    self.log.rooted(rooted_by, derivation.table, roots)
                self.log.flush()
        return counts

    def _named(self, derivation: Derivation, rows: str, parts: str) -> None:
        """Refuse rows of another recipe, or of a partition or inputs the parts do not name."""
        part = DERIVED_PARTS[_family(derivation.table)]
        named = self.lake.execute(
            "SELECT count(*), count(*) FILTER (WHERE n.recipe_key = ? AND EXISTS (SELECT 1"
            f" FROM {parts} p WHERE p.part = {part} AND p.input_digest = n.input_digest))"
            f" FROM {rows} n",
            (derivation.recipe_key,),
        ).fetchall()[0]
        if named[0] != named[1]:
            raise DataError(
                f"{named[0] - named[1]} rows name another recipe, partition or inputs"
                " than those derived; derive each recipe's partitions as they name",
                code="data_layer.derived_rows_mixed",
            )

    def _derived(self, derivation: Derivation, rows: str, parts: str) -> int:
        """Insert, inside the caller's transaction, the rows of each partition the table does not
        hold under its digest yet; the rows written, as the insert counts them."""
        family = _family(derivation.table)
        target, part = f"lake.derived.{derivation.table}", DERIVED_PARTS[family]
        held = f"SELECT DISTINCT recipe_key, {part} AS part, input_digest FROM {target}"
        held += f" WHERE input_digest IN (SELECT input_digest FROM {parts})"
        fresh = f"SELECT n.* FROM {rows} n WHERE NOT EXISTS (SELECT 1 FROM ({held})"
        fresh += " h WHERE h.recipe_key = n.recipe_key AND h.input_digest = n.input_digest"
        fresh += f" AND h.part = {part})"
        order = ", ".join(DERIVED_KEYS[family])
        inserted = self.lake.execute(f"INSERT INTO {target} BY NAME {fresh} ORDER BY {order}")
        return int(inserted.fetchall()[0][0])

    def collect(self, table: str | None = None) -> int:
        """Drop a derived table's partitions whose cache entries no workspace roots (the log's
        cache_root), as one catalog snapshot, and those entries: a cache holds no fact, so a
        partition no workspace reads goes; nothing outside the data root is touched. The rows
        dropped. With no table, every family: each derived
        table the log caches, then the artifact bodies no live reference holds (`collect_unheld`),
        each artifact counted once."""
        if table is None:
            cached = self.log.read("SELECT DISTINCT derived_table FROM cache_entry ORDER BY 1")
            found = sum(self.collect(str(row[0])) for row in cached)
            return found + len(self.collect_unheld())
        with governance.COPYING:
            return self._collect(table)

    def _collect(self, table: str) -> int:
        import pyarrow as pa  # lazy: the optional data extra, as DuckDB

        family = "feature_set_" if table.startswith("feature_set_") else table
        target, run = f"lake.derived.{table}", self.lake.execute
        with self._writer:
            held = run(
                "SELECT 1 FROM duckdb_tables() WHERE database_name = 'lake'"
                " AND schema_name = 'derived' AND table_name = ?",
                (table,),
            ).fetchall()
            if not held:
                return 0
            found = self.log.read(
                "SELECT recipe_key, part, input_digest FROM cache_entry e WHERE derived_table = ?"
                " AND NOT EXISTS (SELECT 1 FROM cache_root r WHERE r.derived_table ="
                " e.derived_table AND r.recipe_key = e.recipe_key AND r.part = e.part"
                " AND r.input_digest = e.input_digest)",
                (table,),
            )
            if not found:
                return 0
            parts = pa.table(
                {
                    name: pa.array([row[at] for row in found], pa.string())
                    for at, name in enumerate(("recipe_key", "part", "input_digest"))
                }
            )
            unrooted = (
                "EXISTS (SELECT 1 FROM collect_parts c WHERE c.recipe_key = t.recipe_key"
                f" AND c.input_digest = t.input_digest AND c.part = {DERIVED_PARTS[family]})"
            )
            self.lake.register("collect_parts", parts)
            try:
                count = run(f"SELECT count(*) FROM {target} t WHERE {unrooted}").fetchall()[0][0]
                if count:
                    run("BEGIN").complete()
                    try:
                        info = canonical({"table": table, "collected": count}).decode()
                        run(
                            "CALL lake.set_commit_message('host', 'derived', extra_info => ?)",
                            (info,),
                        ).complete()
                        run(f"DELETE FROM {target} t WHERE {unrooted}").complete()
                        run("COMMIT").complete()
                    except BaseException:
                        run("ROLLBACK").complete()
                        raise
            finally:
                self.lake.unregister("collect_parts")
            self.log.collect_cached(table)
        return count

    def collect_unheld(self) -> list[str]:
        """`collect_unheld` under this StorePort's writer: the artifacts collected."""
        return collect_unheld(self.root, self.log, self.lake, self._writer)

    def _feature_set(self, table: str, rows: str) -> None:
        """A feature set's wide table as catalog.sql declares it, made on its first partition and
        given a column for each one the rows bring that it does not hold yet, typed as the rows
        type it: a DOUBLE per factor, and the lineage the build stamps on each row. Every change
        is one catalog commit, so the table gains one schema version, never one per column: a
        read scans each schema version's inlined rows, and seventy versions cost a reader's first
        query seconds."""
        run, target = self.lake.execute, f"lake.derived.{table}"
        made = run(
            "SELECT 1 FROM duckdb_tables() WHERE database_name = 'lake' AND schema_name = 'derived'"
            " AND table_name = ?",
            (table,),
        ).fetchall()
        held = {row[0] for row in run(f"DESCRIBE {target}").fetchall()} if made else set()
        added = [
            (column, kind)
            for column, kind in run(
                f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {rows})"
            ).fetchall()
            if column not in held
        ]
        if made and not added:
            return
        self._columns.clear()  # this session's DDL
        run("BEGIN").complete()
        try:
            if not made:
                run(
                    f"CREATE TABLE {target} (recipe_key VARCHAR NOT NULL,"
                    " input_digest VARCHAR NOT NULL, listing_id UUID NOT NULL,"
                    " session_date DATE NOT NULL)"
                ).complete()
                run(f"ALTER TABLE {target} SET PARTITIONED BY (year(session_date))").complete()
                held = {"recipe_key", "input_digest", "listing_id", "session_date"}
            for column, kind in added:
                if column not in held:
                    run(f'ALTER TABLE {target} ADD COLUMN "{column}" {kind}').complete()
            run("COMMIT").complete()
        except BaseException:
            run("ROLLBACK").complete()
            raise

    def maintain_idle(self, now: datetime | None = None) -> None:
        """Run the existing maintenance once per age or write interval at an idle opportunity."""
        now = now or utc_now()
        if not governance.COPYING.acquire(blocking=False):
            return  # leave a machine-wide backup alone
        try:
            if not self._writer.acquire(blocking=False):
                return
            try:
                if self._block is not None:
                    return
                run = self.lake.execute
                ((snapshot, oldest),) = run(
                    "SELECT max(snapshot_id), min(snapshot_time) FROM lake.snapshots()"
                ).fetchall()
                if snapshot is None:
                    return
                prior, checked = self._maintained or (0, oldest)
                if (
                    snapshot - prior < IDLE_MAINTENANCE_WRITES
                    and now - checked < IDLE_MAINTENANCE_AGE
                ):
                    return
                ((threads,),) = run("SELECT current_setting('threads')").fetchall()
                limited = threads > IDLE_MAINTENANCE_THREADS
                if limited:
                    run("SET threads TO ?", (IDLE_MAINTENANCE_THREADS,)).complete()
                try:
                    self.maintain(now)
                    ((latest,),) = run("SELECT max(snapshot_id) FROM lake.snapshots()").fetchall()
                    self._maintained = latest, now
                finally:
                    if limited:
                        run("SET threads TO ?", (threads,)).complete()
            finally:
                self._writer.release()
        finally:
            governance.COPYING.release()

    def maintain(self, now: datetime | None = None) -> dict[str, int]:
        """Keep the catalog compact and its bytes bounded, in one writer hold: flush inlined
        rows, rewrite files past their delete threshold (DERIVED_REWRITE in the derived schema),
        merge adjacent files, expire the snapshots older than KEEP_SNAPSHOTS, and delete the files
        no kept snapshot names once they have waited SETTLE_FILES. Expiry never reaches the oldest
        snapshot whose owed event the log lacks, which the next start's repair reads; when the log
        cannot be read, nothing expires. A retained backup holds its own copy of every file
        (backup.py, its files pool); a running one that finds a listed file deleted is refused by
        name (backup_unreadable) and is run again. The snapshots expired and the files deleted."""
        now = now or utc_now()
        run = self.lake.execute
        with self._writer:
            run(
                "CALL lake.set_option('rewrite_delete_threshold', ?, schema => 'derived')",
                (DERIVED_REWRITE,),
            ).fetchall()
            for step in ("flush_inlined_data", "rewrite_data_files", "merge_adjacent_files"):
                run(f"CALL ducklake_{step}('lake')").fetchall()
            snapshots = [
                (snapshot, at, event["event_id"] if event else None)
                for snapshot, at, message, info in run(
                    "SELECT snapshot_id, snapshot_time, commit_message, commit_extra_info"
                    " FROM lake.snapshots()"
                ).fetchall()
                for event in (owed_events(info) if message == "admission" and info else [None])
            ]
            try:
                held = {
                    row[0]
                    for row in self.log.read(
                        "SELECT event_id FROM event WHERE event_id IN"
                        " (SELECT value FROM json_each(?))",
                        (json.dumps([event for _, _, event in snapshots if event]),),
                    )
                }
            except sqlite3.Error:  # an unreadable log keeps every snapshot
                return {"expired": 0, "deleted": 0}
            expiring = expirable(snapshots, held, now)
            if expiring:
                run("CALL ducklake_expire_snapshots('lake', versions => ?)", (expiring,)).fetchall()
            # A file scheduled for deletion outside this root's data path belongs to the folder
            # a restore moved the catalog from; it is never followed (backup._relocate_catalog).
            posix = r"replace({}, '\', '/')"
            run(
                f"DELETE FROM {METADATA}.ducklake_files_scheduled_for_deletion"
                f" WHERE NOT path_is_relative AND NOT starts_with({posix.format('path')},"
                f" (SELECT {posix.format('value')} FROM {METADATA}.ducklake_metadata"
                "   WHERE key = 'data_path' AND scope IS NULL))"
            ).fetchall()
            deleted = run(
                "CALL ducklake_cleanup_old_files('lake', older_than => ?)", (now - SETTLE_FILES,)
            ).fetchall()
        return {"expired": len(expiring), "deleted": len(deleted)}

    def close(self) -> None:
        """Close the Host's DuckDB session once no call holds it; the log is the caller's. A call
        borrows the session under the writer outside the history's lock, so the close waits for
        it there: DuckDB refuses to close a session a call is using."""
        with self._writer:
            self.lake.close()

    def _registered(self, staged: Staged, source: str) -> None:
        """Refuse an admission naming a lineage the log does not register in its family; its rows
        are read from `source`, a relation."""
        columns = {
            row[0] for row in self.lake.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        named = set() if staged.provider_id is None else {("provider", staged.provider_id)}
        for column, family in LINEAGES.items():
            if column in columns and column != "provider_id":
                found = self.lake.execute(f"SELECT DISTINCT {column} FROM {source}")
                named |= {(family, value) for (value,) in found.fetchall()}
        for family, value in sorted(named):
            known = self.log.read(
                "SELECT 1 FROM definition_lineage WHERE family = ? AND definition_id = ?",
                (family, value),
            )
            if not known:
                raise DataError(
                    f"no {family} lineage has the id {value}; register the {family}, then commit",
                    code="data_layer.lineage_unregistered",
                )
        if "market_id" in columns and staged.table != "market":
            missing = self.lake.execute(
                f"SELECT n.market_id FROM {source} n ANTI JOIN"
                " lake.world.market m USING (market_id) LIMIT 1"
            ).fetchone()
            if missing:
                raise DataError(
                    f"market {missing[0]} is not registered; admit its market record first",
                    code="data_layer.market_unregistered",
                )

    def _lag(self, name: str, version: int) -> tuple[int, str]:
        """The registered lag_rule version's lag and what it counts from (`lag_rule`)."""
        found = self.log.read(
            "SELECT spec FROM definition WHERE family = 'lag_rule' AND name = ? AND version = ?",
            (name, version),
        )
        return lag_rule(json.loads(found[0][0]) if found else {}, name, version)

    def _catalog_commit(
        self, staged: Staged, known: int, lag: tuple[int, str] | None, source: str
    ) -> tuple[int, dict[str, int], int]:
        """One snapshot: the supersession of each world version the rows replace, then the rows and
        the later writes of held values, carrying the event it owes; the rows admitted, those
        dropped and the rewrites recorded, with no snapshot when it admits and records none. A
        batch whose changed rows were all dropped is recorded with no snapshot (`_judged`). Its
        rows are read from `source`, a relation; a KNOWN column dates each one (`_known`)."""
        table, admission = staged.table, staged.admission_id
        key = WORLD_KEYS[table]
        same = " AND ".join(f"t.{k} = n.{k}" for k in key)
        described = self.lake.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        columns = {row[0] for row in described} - {KNOWN}
        carried = KNOWN if len(described) > len(columns) else ""
        target = f"lake.world.{table}"
        run = self.lake.execute
        rule = None if staged.lag is None or lag is None else (*staged.lag, *lag)
        known_rows = self._known(table, source, columns, known, rule, carried)
        run(f"CREATE OR REPLACE TEMP TABLE admitting AS {known_rows}").complete()
        try:
            fates = dict(run("SELECT fate, count(*) FROM temp.admitting GROUP BY ALL").fetchall())
            dropped = {fate: int(count) for fate, count in fates.items() if fate in DROPPED}
            rewritten = int(fates.get(REWRITTEN, 0))
            excluded = ", ".join(("fate", "_held", *([carried] if carried else [])))
            rows = f"SELECT * EXCLUDE ({excluded}) FROM temp.admitting WHERE fate = 'admit'"
            parts, count = self._parts(table, rows)
            if count or staged.lag is not None:
                self._in_order(table, target, same, staged.lag)
            if not count and not rewritten:
                if dropped:
                    self._judged(staged, known, dropped)
                return 0, dropped, 0
            owed = self._owed(staged, known, parts, count, dropped, rewritten)
            self._write(admission, ((table, rows),), owed)
        finally:
            run("DROP TABLE IF EXISTS temp.admitting").complete()
        return count, dropped, rewritten

    def _catalog_results(
        self, staged: Staged | StagedSet, tables: tuple[Staged, ...], known: int
    ) -> int:
        """Write every nonempty table in one snapshot; `_write` checks the named results keys."""
        writes = []
        for batch in tables:
            if not batch.row_count:
                continue
            writes.append((batch.table, f"SELECT * FROM {_staged_rows(self.root, batch)}"))
        count = sum(batch.row_count for batch in tables)
        if count:
            owed = self._owed(staged, known, None, count, {}, 0)
            self._write(staged.admission_id, writes, owed)
        return count

    def _write(
        self, admission: str, writes: Collection[tuple[str, str]], owed: dict[str, Any]
    ) -> None:
        """Put the owed event and every table's writes in the same catalog transaction. A results
        table's keys are checked inside it, so the checks and the writes read one catalog state."""
        block = self._block
        if block is not None:  # the block's transaction, which its end commits or rolls back
            block.writing = True
            self._statements(admission, writes)
            block.writing = False
            block.owed.append(owed)
            return
        run = self.lake.execute
        run("BEGIN").complete()
        try:
            info = {"version": OWED_EVENTS_VERSION, "events": [owed]}
            run(
                "CALL lake.set_commit_message('host', 'admission', extra_info => ?)",
                (canonical(info).decode(),),
            ).complete()
            self._statements(admission, writes)
            run("COMMIT").complete()
        except BaseException:
            run("ROLLBACK").complete()
            raise

    def _statements(self, admission: str, writes: Collection[tuple[str, str]]) -> None:
        """One admission's supersessions and inserts, in its writes' order, in the open
        transaction."""
        run = self.lake.execute
        for table, rows in writes:
            if table in WORLD_KEYS:
                target = f"lake.world.{table}"
                self._supersede(table, target, rows, admission)
                rows = f"SELECT * EXCLUDE (_opening), '{admission}' AS admission_id FROM ({rows})"
            else:
                target = f"lake.results.{table}"
                same = " AND ".join(f"t.{k} = n.{k}" for k in RESULTS_KEYS[table])
                if run(f"SELECT 1 FROM {target} t, ({rows}) n WHERE {same}").fetchone():
                    raise DataError(
                        f"{table} already holds rows of these keys; an artifact's rows are"
                        " admitted once",
                        code="data_layer.result_key_present",
                    )
            order = ", ".join(WORLD_KEYS.get(table) or RESULTS_KEYS[table])
            run(f"INSERT INTO {target} BY NAME {rows} ORDER BY {order}").complete()

    def _supersede(self, table: str, target: str, rows: str, admission: str) -> None:
        """Record the world versions this batch replaces and later writes of held values."""
        run = self.lake.execute
        table_id, key = f"world_table_id('{table}')", WORLD_KEYS[table]
        same = " AND ".join(f"t.{k} = n.{k}" for k in key)
        pair = " AND ".join(f"o.{k} = c.{k}" for k in key)
        run(  # once, by its earliest replacement: an interval's pair replaces it once
            f"INSERT INTO lake.world.superseded SELECT {table_id}, t.version_id,"
            f" min(n.known_asof), '{admission}' FROM {target} t, ({rows}) n"
            f" WHERE {same} AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s"
            f" WHERE s.table_id = {table_id} AND s.version_id = t.version_id)"
            " GROUP BY t.version_id"
        ).complete()
        run(
            f"INSERT INTO lake.world.superseded SELECT {table_id}, o.version_id,"
            f" c.known_asof, '{admission}' FROM ({rows}) o, ({rows}) c"
            f" WHERE {pair} AND o._opening AND NOT c._opening"
        ).complete()
        if table in TIMED:
            run(
                f"INSERT INTO lake.world.rewritten SELECT {table_id}, _held,"
                f" {OBSERVED}, '{admission}' FROM temp.admitting"
                f" WHERE fate = '{REWRITTEN}'"
            ).complete()

    def _in_order(self, table: str, target: str, same: str, lag: tuple[str, int] | None) -> None:
        """Refuse rows known before a version of their key already admitted: a key's versions
        are admitted in their known_asof order, so each supersedes the one before it. Refuse a
        legacy batch any of whose keys world holds under another lag rule or version: those
        rows' known_asof was derived by it, and an upgrade step re-derives them first."""
        moved = "0"
        if lag is not None:
            moved = (
                "count(*) FILTER (WHERE t.lag_rule_version IS NOT NULL AND (t.lag_rule"
                " IS DISTINCT FROM n.lag_rule OR t.lag_rule_version IS DISTINCT FROM"
                " n.lag_rule_version))"
            )
        late, other = self.lake.execute(
            "SELECT count(*) FILTER (WHERE n.fate = 'admit' AND n.known_asof < t.known_asof),"
            f" {moved} FROM {target} t, temp.admitting n WHERE {same}"
        ).fetchone()
        if other:
            raise DataError(
                f"{other} {table} rows' keys are held under another lag rule or version than"
                f" {lag}; re-derive the held rows under it first, or admit under theirs",
                code="data_layer.lag_rule_version_moved",
            )
        if late:
            raise DataError(
                f"{late} {table} rows would be known before a version of their key already"
                " admitted; admit a key's versions in their known_asof order",
                code="data_layer.admission_out_of_order",
            )

    def _judged(self, staged: Staged, known: int, dropped: dict[str, int]) -> None:
        """Record an admission whose every changed row was dropped, with each drop's count, once:
        a retried commit of it finds its event. Its event names the snapshot it was judged against:
        the newest committed one, recorded now and durably, before its staging goes, or, judged
        against a block's own writes, the block's, whose commit carries the event in its order
        (`admitting`)."""
        if self.log.read(
            "SELECT 1 FROM event WHERE type = 'admission.committed' AND subject_id = ?",
            (staged.admission_id,),
        ):
            return
        owed = self._owed(staged, known, [], 0, dropped, 0)
        if self._block is not None and self._block.owed:
            self._block.owed.append(owed)
            return
        snapshot = self.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
        row = {column: owed[column] for column in ENVELOPE if column in owed and column != "seq"}
        attrs = json.loads(owed["attrs"]) | {"catalog_snapshot": snapshot[0]}
        row["attrs"] = canonical(attrs).decode()
        self.log.append(row, durable=True)

    def _known(
        self,
        table: str,
        source: str,
        columns: set[str],
        known: int,
        rule: tuple[str, int, int, str] | None,
        carry: str = "",
    ) -> str:
        """Every staged row (with its `carry` column, if named), conformed to its table's catalog
        types, with its version_id (a UUIDv7
        the engine mints), its known_at, its known_asof and its fate, as a query: `admit`, `same`
        (its content equals its key's newest version's, column by column, so however a writer
        stages one fact it is one version), `held_opening` (the open version of an interval world
        already holds), or a DROPPED reason. A row may carry `retracted` to withdraw its key's
        fact; a column it leaves out is null, and `retracted` false.

        `known` is the commit's clock, or, where `carry` is KNOWN, each row's own observed time,
        which then stands for it in every rule below. An observed row's times are both `known`.
        A session's recorded source observation takes
        precedence; a first version generated by a named, versioned calendar rule is reference
        data known before every cutoff, with no invented observation time. A later correction
        has its recorded observation time or this admission's clock.
        A legacy row, staged under a lag `rule` (its
        name, version, seconds and what they count from), names the rule and version; its known_at
        is empty and its known_asof is its publication time, or its lag after the close of its
        market's first session on or after its valid date, or after its valid date's start (UTC)
        under a rule that counts from it, the date its source recorded it observed as of, and no
        later than this admission; the market is the row's own listing's, known only while that
        listing has had one market.
        An UNMARKETED row (a document) is known its lag after its
        own valid time, its availability. A legacy row of a key world already holds is a change
        observed now: known no earlier than its admission or the version it replaces, so no read
        before then sees it; but a legacy document over a legacy version (its known_at empty) is
        another 0.2.7 fact with its own availability, known then, never before the version it
        replaces. A legacy interval with an end, for a key world does not hold yet, is
        written as its two versions: the open one known from its start, the closing one from its
        end; a delisted provider mapping's opening state is ACTIVE. In a TIMED table, a held
        key is superseded only by a row its source wrote after the newest version's value was
        last written (its ordering time, `newest`), and a later write
        of that same value is REWRITTEN."""
        at = f"make_timestamptz({f'n.{carry}' if carry == KNOWN else known}::BIGINT)"
        key = WORLD_KEYS[table]
        declared = self._declared(table, columns)
        conformed = ", ".join(
            f"CAST({name} AS {kind}) AS {name}"
            if name in columns
            else f"{'false' if name == 'retracted' else 'NULL'}::{kind} AS {name}"
            for name, kind in declared.items()
        )
        carried = f", {carry}" if carry else ""
        rows = f"SELECT {conformed}{carried}, false AS _opening FROM {source}"
        interval = LEGACY_VALID[table] if rule is not None else ()
        if len(interval) == 2 and interval[0] in columns:  # an interval's open version, beside
            end = interval[0]  # its closing one
            opening = f"NULL::{declared[end]} AS {end}, true AS _opening"
            if table == "provider_symbol" and "lifecycle_state" in columns:
                opening += (
                    ", CASE WHEN lifecycle_state = 'DELISTED' THEN 'ACTIVE'"
                    " ELSE lifecycle_state END AS lifecycle_state"
                )
            rows = (
                f"SELECT * FROM ({rows}) UNION ALL SELECT * REPLACE ({opening})"
                f" FROM ({rows}) WHERE {end} IS NOT NULL"
            )
        held = f" LEFT JOIN {newest(table)} p ON " + " AND ".join(f"p.{k} = n.{k}" for k in key)
        timed = table in TIMED
        written = f"n.{OBSERVED}"
        value = " AND ".join(
            f"p.{name} IS NOT DISTINCT FROM n.{name}"
            for name in declared
            if name not in key and name not in SOURCE
        )
        fate = "CASE"
        if timed:
            fate += f" WHEN {value} AND {written} > p.ordered_at THEN '{REWRITTEN}'"
        fate += (
            f" WHEN {value} THEN 'same'"
            " WHEN n._opening AND p.version_id IS NOT NULL THEN 'held_opening'"
        )
        if timed:
            fate += (
                f" WHEN {written} < p.ordered_at THEN 'older'"
                f" WHEN {written} = p.ordered_at THEN 'equal_time'"
            )
        fate += " ELSE 'admit' END AS fate, p.version_id AS _held, uuidv7() AS version_id"
        if rule is None:
            if table == "session":
                observed = f"coalesce(n.source_observed_at, {at})"
                reference = (
                    "p.version_id IS NULL AND n.source_observed_at IS NULL"
                    " AND n.calendar_origin IS NOT NULL AND n.calendar_version IS NOT NULL"
                )
                return (
                    f"SELECT n.*, CASE WHEN {reference} THEN NULL::TIMESTAMPTZ ELSE {observed}"
                    f" END AS known_at, CASE WHEN {reference} THEN '-infinity'::TIMESTAMPTZ"
                    f" ELSE {observed} END AS known_asof, {fate} FROM ({rows}) n{held}"
                )
            return f"SELECT n.*, {at} AS known_at, {at} AS known_asof, {fate} FROM ({rows}) n{held}"
        valid = ", ".join(f"n.{c}" for c in LEGACY_VALID[table] if c in columns)
        name, version, lag, start = rule
        quoted = name.replace("'", "''")
        named = f"'{quoted}' AS lag_rule, {version}::INTEGER AS lag_rule_version"
        if table in UNMARKETED:
            derived = f"coalesce({valid}) + to_seconds({lag})"
            asof = (  # over a legacy version, a 0.2.7 fact with its own availability as well
                f"CASE WHEN p.version_id IS NULL THEN {derived}"
                f" WHEN p.known_at IS NULL THEN greatest({derived}, p.known_asof)"
                f" ELSE greatest({derived}, {at}, p.known_asof) END"
            )
            return (
                f"SELECT n.*, NULL::TIMESTAMPTZ AS known_at, {asof} AS known_asof, {fate},"
                f" {named} FROM ({rows}) n{held}"
            )
        derived = f"a.close_utc + to_seconds({lag})"
        if start == "valid":  # observed by its admission at the latest, never at a future date
            derived = (
                f"least(make_timestamptz(epoch_us(CAST(coalesce({valid}) AS TIMESTAMP)))"
                f" + to_seconds({lag}), {at})"
            )
        if "published_at" in columns:
            derived = f"coalesce(n.published_at, {derived})"
        market, joined = "n.market_id", ""
        if table != "listing":
            market = "l.market_id"
            joined = (  # a listing that changed market has no market for a date before the change
                " LEFT JOIN (SELECT listing_id, any_value(market_id) AS market_id"
                " FROM lake.world.listing"
                " WHERE NOT retracted GROUP BY listing_id HAVING count(DISTINCT market_id) = 1) l"
                " ON l.listing_id = n.listing_id"
            )
        asof = (
            f"CASE WHEN p.version_id IS NULL THEN {derived}"
            f" ELSE greatest({derived}, {at}, p.known_asof) END"
        )
        return (
            f"SELECT n.*, NULL::TIMESTAMPTZ AS known_at, {asof} AS known_asof, {fate},"
            f" {named} FROM ({rows}) n{joined} ASOF LEFT JOIN {_current('session')} a"
            f" ON a.market_id = {market} AND coalesce({valid}) <= a.session_date{held}"
        )

    def _declared(self, table: str, columns: set[str]) -> dict[str, str]:
        """A world table's content and SOURCE columns with their catalog types, as the engine
        holds them, described once a session; refused by name when the rows carry a column the
        table does not hold."""
        declared = self._columns.get(table)
        if declared is None:
            (described,) = self.lake.execute(
                "SELECT map(list(column_name), list(column_type))"
                f" FROM (DESCRIBE lake.world.{table})"
            ).fetchone()
            declared = {name: kind for name, kind in described.items() if name not in ADMITTED}
            self._columns[table] = declared
        foreign = sorted(columns - set(declared))
        if foreign:
            raise DataError(
                f"{table} holds no column {', '.join(foreign)}; send only its own"
                f" ({', '.join(declared)})",
                code="data_layer.admission_column_foreign",
            )
        return declared

    def _parts(self, table: str, rows: str) -> tuple[list[dict[str, Any]], int]:
        """Each part the rows fall in, with the range of their known_asof there, and how many rows
        there are; refused by name when a legacy row found no session to count its lag from."""
        column = PARTITION.get(table)
        part = f"CAST(year(CAST({column} AS DATE)) AS VARCHAR)" if column else "'all'"
        clock = "epoch_us(known_asof)"
        if table == "session":
            clock = (
                "CASE WHEN known_asof = '-infinity'::TIMESTAMPTZ THEN -9223372036854775808"
                " ELSE epoch_us(known_asof) END"
            )
        found = self.lake.execute(
            f"SELECT {part} AS part, min({clock}), max({clock}),"
            f" count(known_asof), count(*) FROM ({rows})"
            " GROUP BY ALL HAVING count(*) > 0 ORDER BY part"
        ).fetchall()
        count = sum(row[4] for row in found)
        unknown = count - sum(row[3] for row in found)
        if unknown:
            raise DataError(
                f"{unknown} legacy {table} rows have no session close to count from: their listing"
                " is not in world.listing, or has had more than one market so its market then is"
                " unknown, or that market has no session on or after their valid date",
                code="data_layer.legacy_anchor_missing",
            )
        parts = [
            {"part": p, "min_known_asof_us": lo, "max_known_asof_us": hi} for p, lo, hi, *_ in found
        ]
        return parts, count

    def _owed(
        self,
        staged: Staged | StagedSet,
        known: int,
        parts: list[dict[str, Any]] | None,
        count: int,
        dropped: dict[str, int],
        rewritten: int,
    ) -> dict[str, Any]:
        """The `admission.committed` event the catalog commit carries, for the `count` rows it
        admits, those it `dropped` and the `rewritten` writes it records; a world admission that
        admits no row touches no part."""
        whole = [{"part": "all", "min_known_asof_us": known, "max_known_asof_us": known}]
        attrs: dict[str, Any] = {
            "row_count": count,
            "known_us": known,
            "parts": whole if parts is None else parts,
        }
        if isinstance(staged, StagedSet):
            attrs |= {
                "provider_id": None,
                "dataset": "results",
                "tables": {batch.table: batch.row_count for batch in staged.tables},
            }
            artifacts = sorted(
                {artifact for batch in staged.tables for artifact in batch.artifacts}
            )
        else:
            attrs |= {"provider_id": staged.provider_id, "dataset": staged.table}
            if staged.lag is not None:
                attrs |= {"lag_rule": staged.lag[0], "lag_rule_version": staged.lag[1]}
            artifacts = list(staged.artifacts)
        if artifacts:
            attrs["artifact_ids"] = artifacts
        if dropped:
            attrs["dropped"] = dropped
        if rewritten:
            attrs[REWRITTEN] = rewritten
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


def _staged_rows(root: Path, staged: Staged) -> str:
    """The relation of a staged admission's rows: its Parquet file under staging."""
    parquet = root / "staging" / staged.admission_id / f"{staged.table}.parquet"
    return f"read_parquet('{sql_path(parquet)}')"


@contextmanager
def stage(
    root: Path,
    table: str | Mapping[str, Any],
    provider_id: str | None = None,
    rows: Any = None,
    *,
    lag: tuple[str, int] | None = None,
    floor_bytes: int = FREE_FLOOR_BYTES,
) -> Iterator[Staged | StagedSet]:
    """Write one admission's rows, an Arrow table or record batch reader, under staging as they
    came, in Arrow from the caller to the Parquet; the commit request it yields
    is all the writer needs. `provider_id` is a world table's provider lineage in the log, and None
    for a results table, whose rows' artifact_id is their lineage. A mapping of results table names
    to Arrow rows stages them together, with no provider or lag. Its producer holds
    the admission's lock until the block ends, so a Host that starts meanwhile leaves it: commit
    it inside the block. A legacy batch, rows 0.2.7 kept with no knowledge time, names its
    source's lag rule as `lag=(name, version)`. Refused by name below the free-space floor, and
    as `admissible` refuses.
    """
    short = free_floor_shortfall(root, floor_bytes)
    if short:
        raise DataError(
            f"the data root's disk is {short} bytes below its {floor_bytes}-byte free-space floor,"
            " kept for the log, its repair and cleanup; free space on that disk first",
            code="data_layer.below_free_floor",
        )
    admission = uuid7()  # first: the rows and the staging folder carry it
    if isinstance(table, str):
        sources = ((table, rows),)
    else:
        if not table or provider_id is not None or rows is not None or lag is not None:
            raise TypeError("Stage a results set with a nonempty mapping of table names to rows.")
        sources = tuple(table.items())
    prepared = [
        (name, admissible(name, provider_id, source, legacy=lag is not None))
        for name, source in sources
    ]
    import pyarrow.parquet as pq  # lazy: the optional data extra, as DuckDB

    lock = root / "staging" / f"{admission}.lock"
    held = hold(lock)  # before the folder, so no start sees the folder unheld
    if held is None:
        raise DataError(
            "a starting Host collected this admission's staging before its producer held it;"
            " stage the rows again",
            code="data_layer.staging_taken",
        )
    try:
        folder = root / "staging" / admission
        folder.mkdir()
        tables = []
        for name, data in prepared:
            pq.write_table(data, folder / f"{name}.parquet")
            artifacts = (
                tuple(sorted(data.column("artifact_id").unique().to_pylist()))
                if name in RESULTS_KEYS
                else ()
            )
            tables.append(Staged(admission, name, provider_id, data.num_rows, lag, artifacts))
        yield tables[0] if len(tables) == 1 else StagedSet(admission, tuple(tables))
    finally:
        drop(held, lock)


def lag_rule(spec: Mapping[str, Any], name: str, version: int) -> tuple[int, str]:
    """A lag_rule version's spec read: the lag it adds, in seconds, and what it counts from: a
    session's `close` (the default), or the start of the row's `valid` date (UTC); refused by name
    when it names neither or no whole lag of 0 or more."""
    seconds, start = spec.get("lag_seconds"), spec.get("from", "close")
    if type(seconds) is not int or seconds < 0 or start not in ("close", "valid"):
        raise DataError(
            f"lag_rule {name} v{version} is not registered with whole lag_seconds of 0 or"
            " more from a close or a valid date; register the rule, then commit the batch",
            code="data_layer.lag_rule_unknown",
        )
    return seconds, start


def newest(table: str) -> str:
    """Each key's newest version in a world table, every column, as a subquery. A TIMED table's
    also carries `ordered_at`: its value's last write by its source, the later of the version's own
    and each world.rewritten write of it, -infinity when there is none, so a row with a source time
    orders after a version without one."""
    table_id = f"world_table_id('{table}')"
    current = (
        f"SELECT x.* FROM lake.world.{table} x WHERE NOT EXISTS (SELECT 1 FROM"
        f" lake.world.superseded s WHERE s.table_id = {table_id}"
        " AND s.version_id = x.version_id)"
    )
    if table not in TIMED:
        return f"({current})"
    return (
        f"(SELECT c.*, coalesce(greatest(c.{OBSERVED}, w.latest), '-infinity'::TIMESTAMPTZ)"
        f" AS ordered_at FROM ({current}) c LEFT JOIN (SELECT version_id, max({OBSERVED})"
        f" AS latest FROM lake.world.rewritten WHERE table_id = {table_id} GROUP BY ALL) w"
        " ON w.version_id = c.version_id)"
    )


def feature_table(feature_set_id: str) -> str:
    """A feature set's table in `derived`, named by its lineage's id and never its name, so a
    renamed feature set keeps every cached partition: feature_set_ and the id's hex digits."""
    return "feature_set_" + feature_set_id.removeprefix("blake3:").replace("-", "")


def admissible(table: str, provider_id: str | None, rows: Any, *, legacy: bool = False) -> Any:
    """The rows as one Arrow table, each with its provider's lineage where its table keys one, so
    a renamed provider moves no version.

    Refused by name: a table StorePort does not admit, a legacy batch for a table with no valid
    date to count a lag from, a TIMED row with no source time (OBSERVED), which could not be
    ordered against a held value, and two rows of one key, since DuckLake keeps no keys.
    """
    key = WORLD_KEYS.get(table) or RESULTS_KEYS.get(table)
    if key is None:
        raise DataError(
            f"{table} is not a table StorePort admits; it admits"
            f" {', '.join([*WORLD_KEYS, *RESULTS_KEYS])}",
            code="data_layer.world_table_not_admitted",
        )
    if (provider_id is None) is (table in WORLD_KEYS):
        raise DataError(
            f"a {table} admission names a provider lineage for a world table and none for a results"
            " table, whose rows' artifact_id is their lineage; stage it with"
            f" {'its provider' if table in WORLD_KEYS else 'no provider'}",
            code="data_layer.admission_lineage_invalid",
        )
    if legacy and table not in LEGACY_VALID:
        raise DataError(
            f"{table} rows have no valid date a lag rule counts from; admit them as observed",
            code="data_layer.legacy_table_unanchored",
        )
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    data = rows.read_all() if isinstance(rows, pa.RecordBatchReader) else rows
    if table in TIMED and (OBSERVED not in data.column_names or data.column(OBSERVED).null_count):
        raise DataError(
            f"{table} rows need {OBSERVED}, the source's own last write of the value: without"
            " it a row could supersede a value its source wrote later; send it in every row",
            code="data_layer.admission_source_time_missing",
        )
    if "provider_id" in key:
        lineage = pa.repeat(pa.scalar(provider_id, pa.string()), data.num_rows)
        at = data.schema.get_field_index("provider_id")
        data = (
            data.set_column(at, "provider_id", lineage)
            if at >= 0
            else data.append_column("provider_id", lineage)
        )
    repeated = _repeated(data, key)
    if repeated is not None:
        raise DataError(
            f"{table} {dict(zip(key, repeated, strict=True))} appears twice in one admission;"
            " send one version of a key",
            code="data_layer.admission_key_repeated",
        )
    return data


def _repeated(data: Any, key: tuple[str, ...]) -> tuple[Any, ...] | None:
    """The first key a later row repeats, in row order; None when every key is once."""
    keys = data.select(list(key))
    if key != ("doc_id",):
        import pyarrow.acero  # noqa: F401  # lazy: the grouping engine for general admissions

        if keys.group_by(list(key)).aggregate([]).num_rows == keys.num_rows:
            return None
    seen: set[tuple[Any, ...]] = set()
    for values in zip(*(keys.column(name).to_pylist() for name in key), strict=True):
        if values in seen:
            return values
        seen.add(values)
    return None


def as_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """The rows as columns: every column any row has, empty where a row lacks it. Arrow's own
    `from_pylist` takes its columns from the first row and would drop a later row's field."""
    names = dict.fromkeys(name for row in rows for name in row)
    return {name: [row.get(name) for row in rows] for name in names}


def repair(log: HostLog, lake: Any, after: int = 0) -> list[str]:
    """Append each event a catalog snapshot after `after` carried and the log lacks; returns their
    ids. The start reads every snapshot; a commit only those after the last it closed.

    An owed event keeps every field its writer made, its id, version, workspace, actor, trace and
    time included; only its recording time is new, after the moment it carries, and its
    attributes name the snapshot. Running it again adds none. Only an admission's commit owes an
    event; a derived write's commit info only describes its rows.
    """
    carried = lake.execute(  # the changes table alone: lake.snapshots() lists every snapshot whole
        f"SELECT snapshot_id, commit_extra_info FROM {CHANGES}"
        " WHERE commit_message = 'admission' AND commit_extra_info IS NOT NULL"
        " AND snapshot_id > ? ORDER BY snapshot_id",
        (after,),
    ).fetchall()
    added = []
    for snapshot, info in carried:
        for owed in owed_events(info):
            if log.read("SELECT 1 FROM event WHERE event_id = ?", (owed["event_id"],)):
                continue
            attrs = json.loads(owed["attrs"])
            row = {c: owed[c] for c in ENVELOPE if c in owed and c != "seq"}
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
    take it meanwhile, settles each array body a crash left staged under blobs/ (placed when its
    profile was committed, else collected; `arrays.settle`), and collects every file under data/
    the catalog does not name.
    """
    report: dict[str, list[str]] = {"repaired": repair(log, lake), "collected": [], "waiting": []}
    removed = remove_bodies(root, log, lake)  # what a collection queued before a crash
    report["collected"] += removed["collected"]
    report["waiting"] += removed["waiting"]
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
    from alphalattice.kernel.data_layer.arrays import settle  # lazy: numpy and BLAKE3

    settled = settle(root, log.read, lambda handle: delete(root, log, handle))
    report["placed"] = settled["placed"]
    report["collected"] += settled["collected"]
    report["waiting"] += settled["waiting"]
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


def collect_unheld(
    root: Path, log: HostLog, lake: Any, writer: AbstractContextManager[Any]
) -> list[str]:
    """Collect each unheld artifact body older than the configured window, `governance.UNHELD`
    by default, holding `governance.COPYING` and then `writer`. Its
    `artifact.collected` and its body's place in the deletion queue are decided in one log unit,
    so no hold slips between the read and the event, and committed before the body goes, so a
    crash over-retains and the next start finishes it; an inline body goes with the event. The
    artifacts collected."""
    with governance.COPYING, writer:
        days = process_config().settings.retention_days
        tick = timedelta(microseconds=1)
        age = governance.UNHELD // tick if days is None else days * (timedelta(days=1) // tick)
        before = max(0, log.clock() - age)
        with log.unit():
            found = governance.unheld(log.read, before)
            for artifact, storage in found:
                log.record("artifact.collected", ("artifact", artifact), {})
                if storage in governance.BODY_HANDLES:
                    folder, kind = governance.BODY_HANDLES[storage]
                    log.queue_deletion(f"{folder}/{artifact}", kind, "collected")
        log.flush()
        remove_bodies(root, log, lake)
        import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

        inputs.drop_views(root, log.read)  # a view's links to a collected input's bytes
    return [artifact for artifact, _ in found]


def remove_bodies(root: Path, log: HostLog, lake: Any) -> dict[str, list[str]]:
    """Remove the bodies of collected artifacts the deletion queue names: each blob, and the
    rows of every `results/<artifact_id>` from each results table in one catalog commit. Run
    under the writer, or at the start before any; the handles removed, and those still held."""
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    queued = log.read(
        "SELECT handle, kind FROM deletion_queue WHERE kind IN ('blob', 'rows') ORDER BY handle"
    )
    report: dict[str, list[str]] = {"collected": [], "waiting": []}
    for handle in (str(row[0]) for row in queued if row[1] == "blob"):
        report["collected" if delete(root, log, handle) else "waiting"].append(handle)
    rows = [str(row[0]) for row in queued if row[1] == "rows"]
    if not rows:
        return report
    tables = lake.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'lake'"
        " AND schema_name = 'results' ORDER BY table_name"
    ).fetchall()
    ids = pa.table({"artifact_id": pa.array([Path(handle).name for handle in rows], pa.string())})
    lake.register("collected_ids", ids)
    try:
        lake.execute("BEGIN").complete()
        try:
            info = canonical({"collected": len(rows)}).decode()
            lake.execute(
                "CALL lake.set_commit_message('host', 'collect', extra_info => ?)", (info,)
            ).complete()
            for (name,) in tables:
                lake.execute(
                    f"DELETE FROM lake.results.{name} WHERE artifact_id IN"
                    " (SELECT artifact_id FROM collected_ids)"
                ).complete()
            lake.execute("COMMIT").complete()
        except BaseException:
            lake.execute("ROLLBACK").complete()
            raise
    finally:
        lake.unregister("collected_ids")
    for handle in rows:
        log.dequeue_deletion(handle)
    report["collected"] += rows
    return report


def _family(table: str) -> str:
    """A derived table's family in DERIVED_KEYS: a feature set's by its prefix."""
    return "feature_set_" if table.startswith("feature_set_") else table


def _current(table: str) -> str:
    """A world table's current versions, as a subquery: those not withdrawn and not replaced."""
    return (
        f"(SELECT * FROM lake.world.{table} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1"
        f" FROM lake.world.superseded s WHERE s.table_id = world_table_id('{table}')"
        " AND s.version_id = x.version_id))"
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
