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
every world version they replace and carries in its commit
info the whole envelope of the `admission.committed` it owes; then `repair`
appends that event, durably, and the staging folder is deleted. `start` runs the same `repair`
when the Host opens, so the normal path and crash recovery are one path, and collects what no
snapshot or write owns. Every file leaves through `delete`: one another process holds open waits
in deletion_queue and is retried at each start, and nothing a live_ref holds is ever deleted.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, NamedTuple

from alphalattice.control.workspace_runtime.storage.capacity import (
    FREE_FLOOR_BYTES,
    free_floor_shortfall,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import ENVELOPE, canonical
from alphalattice.kernel.data_layer.log import HostLog, drop, envelope, hold, uuid7
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.root import (
    THREADS,
    catalog,
    catalog_matches,
    load_extensions,
    schema_text,
    sql_path,
    statements,
)
from alphalattice.kernel.data_layer.telemetry import usage

KINDS: Final = {"staging": "staging", "blobs": "blob", "data": "file"}
"""The deletion queue's kind for each top folder a deleted file may live in."""
WORLD_KEYS: Final = {
    "listing": ("listing_id",),
    "provider_symbol": ("provider_id", "symbol", "valid_from"),
    "universe_member": ("universe_id", "listing_id", "valid_from"),
    "sector": ("scheme_id", "listing_id", "valid_from"),
    "raw_bar": ("provider_id", "listing_id", "session_date"),
    "corporate_action": ("provider_id", "listing_id", "effective_date", "kind", "action_id"),
    "session": ("market", "session_date"),
    "evidence_doc": ("doc_id",),
}
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
TABLE_IDS: Final = {table: definition_id("world_table", table) for table in WORLD_KEYS}
"""Each world table's lineage id, as the seed registers it: world.superseded names a table by it."""
LINEAGES: Final = {
    "provider_id": "provider",
    "scheme_id": "sector_scheme",
    "universe_id": "universe",
}
"""The world columns that name a lineage, with its family: a row names a registered one."""
RESULTS_KEYS: Final = {"execution_outcome": ("artifact_id", "listing_id", "formation_session")}
"""The results tables StorePort admits, with their keys. A results row is an artifact's body,
written once: a key already in the catalog is refused. Its lineage is its artifact_id, not an
outside provider: its admission names no provider and lists the artifacts it holds."""
DERIVED_KEYS: Final = {
    "feature_set_": ("recipe_key", "input_digest", "session_date", "listing_id"),
    "training_sample": ("recipe_key", "input_digest", "row_no"),
    "factor_cov": ("recipe_key", "input_digest", "as_of", "factor_i", "factor_j"),
    "factor_loading": ("recipe_key", "input_digest", "as_of", "listing_id", "factor"),
    "specific_risk": ("recipe_key", "input_digest", "as_of", "listing_id"),
}
"""The derived caches StorePort admits, with their keys: a feature set's table by its prefix
(feature_table). A cache is rebuildable and no row references it, so its key may hold a digest;
its rows are written only by `StorePort.derive`, and the log's cache_entry says which are valid."""
DERIVED_PARTS: Final = {
    "feature_set_": (
        "strftime(CAST(session_date AS DATE), '%Y-%m') || '/' || CAST(listing_id AS VARCHAR)"
    ),
    "training_sample": "CAST(year(CAST(session_date AS DATE)) AS VARCHAR)",
    "factor_cov": "CAST(CAST(as_of AS DATE) AS VARCHAR)",
    "factor_loading": "CAST(CAST(as_of AS DATE) AS VARCHAR)",
    "specific_risk": "CAST(CAST(as_of AS DATE) AS VARCHAR)",
}
"""Each derived table's partition of a row, as cache_entry's `part` names it: a year, a month and
listing (a feature set's: a build's batch writes whole listings, and a day's append moves one
month of each), or an as-of date."""
KEEP_SNAPSHOTS: Final = timedelta(days=7)
"""How long the catalog keeps a snapshot: the plan's snapshot rule (0.2.md, 0.2.8 decision 2)."""
SETTLE_FILES: Final = timedelta(days=3)
"""How long a file no kept snapshot names waits before it is deleted, so a read that listed it
before its snapshot expired still finds it: DuckLake's own guidance, a few days."""
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


class Committed(NamedTuple):
    """What one commit did: its snapshot, None when it wrote nothing; the rows it admitted; the
    changed rows of held keys it dropped, by DROPPED's reasons; and the later writes of held values
    it recorded (REWRITTEN)."""

    snapshot: int | None
    rows: int
    dropped: dict[str, int]
    rewritten: int


def expirable(
    snapshots: list[tuple[int, datetime, str | None]], held: set[str], now: datetime
) -> list[int]:
    """The snapshots maintenance may expire, of (id, time, its admission's event id or None):
    those older than KEEP_SNAPSHOTS, never the newest, and none from the oldest admission whose
    event the log does not hold (`held`) on, since the next start's repair reads its commit."""
    owed = min((s for s, _, event in snapshots if event and event not in held), default=None)
    newest = max((s for s, _, _ in snapshots), default=None)
    return sorted(
        s
        for s, at, _ in snapshots
        if at < now - KEEP_SNAPSHOTS and s != newest and (owed is None or s < owed)
    )


class StorePort:
    """The Host's one writer of the catalog; open it with `StorePort.open`."""

    def __init__(self, root: Path, log: HostLog, lake: Any):
        """Hold the data root, its log and the Host's DuckDB session; `open` makes one."""
        self.root, self.log, self.lake = root, log, lake
        self._writer = threading.Lock()
        self.started: dict[str, list[str]] = {}

    @classmethod
    def open(cls, root: Path, log: HostLog) -> StorePort:
        """Attach the catalog and the reading surface's world macros, then run `start`. The log
        is never attached: this process writes it through Python's sqlite3, and a second SQLite
        library closing its descriptor would release the writer's locks on the file."""
        import duckdb  # lazy: DuckDB is the optional data extra; only a Host with data opens this

        lake = duckdb.connect(  # in memory: the catalog is attached to it
            ":memory:",
            config={
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
                "threads": THREADS,
            },
        )
        load_extensions(lake)
        catalog(root)  # made at the store's first open, not with the root
        machine, data = sql_path(root / "machine"), sql_path(root / "data")
        for sql in (
            f"ATTACH 'ducklake:sqlite:{machine}/catalog.sqlite' AS lake"
            f" (DATA_PATH '{data}/', DATA_INLINING_ROW_LIMIT 1000)",
            "SET TimeZone = 'UTC'",
        ):
            lake.execute(sql).complete()
        catalog_matches(lake)  # before the views, which name its columns
        for sql in statements(schema_text("views.sql")):
            if "log." not in sql:
                lake.execute(sql).complete()
        port = cls(root, log, lake)
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

        with self._writer:
            received_us = self.log.clock()
            batch = pa.Table.from_pylist(
                [
                    row | {"received_us": received_us, "record_index": i}
                    for i, row in enumerate(rows)
                ]
            )
            self.lake.register("agent_batch", batch)
            try:
                self.lake.execute("BEGIN TRANSACTION").complete()
                before = self.lake.execute("SELECT count(*) FROM lake.telemetry.event").fetchone()[
                    0
                ]
                self.lake.execute(
                    "INSERT INTO lake.telemetry.event BY NAME"
                    " SELECT DISTINCT b.* FROM agent_batch b WHERE NOT EXISTS"
                    " (SELECT 1 FROM lake.telemetry.event e WHERE e.event_id = b.event_id)"
                ).complete()
                after = self.lake.execute("SELECT count(*) FROM lake.telemetry.event").fetchone()[0]
                self.lake.execute("COMMIT").complete()
                return int(after - before)
            except BaseException:
                self.lake.execute("ROLLBACK").complete()
                raise
            finally:
                self.lake.unregister("agent_batch")

    def agent_telemetry(
        self, host: str, session: str, *, workspace_id: str
    ) -> list[dict[str, Any]]:
        """Read this workspace's exact vendor and conversation/session, in event order."""
        with self._writer:
            result = self.lake.execute(
                "SELECT * FROM agent_telemetry WHERE workspace_id = ? AND vendor = ?"
                " AND (conversation_id = ? OR session_id = ? OR parent_agent_id = ?)"
                " ORDER BY received_us, record_index",
                [workspace_id, host, session, session, session],
            )
            columns = [column[0] for column in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def unplaced_telemetry(self, *, workspace_id: str) -> int:
        """Count only this workspace's observations without a conversation or session."""
        with self._writer:
            return int(
                self.lake.execute(
                    "SELECT count(*) FROM agent_telemetry WHERE workspace_id = ? AND unplaced",
                    [workspace_id],
                ).fetchone()[0]
            )

    def trial_agent_usage(self, trial: str, *, workspace_id: str) -> dict[str, Any]:
        """The observed native sums a trial closes with, with no invented zero on absence."""
        with self._writer:
            result = self.lake.execute(
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

    def commit(self, staged: Staged) -> Committed:
        """Commit one staged admission as one catalog snapshot, then its owed event. A world row
        already its key's newest version is not admitted again, so a repeated batch adds nothing
        and takes no snapshot. The event is durable, so it is on disk when this returns. Refused
        by name: a lag rule the log does not hold, legacy rows with no session to count from,
        results keys already admitted, and a provider, scheme or universe no lineage in the log
        registers."""
        with self._writer:
            self._registered(staged)
            lag = None if staged.lag is None else self._lag_seconds(*staged.lag)
            count, dropped, rewritten = self._catalog_commit(staged, self.log.clock(), lag)
            repair(self.log, self.lake)
            delete(self.root, self.log, f"staging/{staged.admission_id}")
            if not count and not rewritten:
                return Committed(None, 0, dropped, 0)
            found = self.log.read(
                "SELECT catalog_snapshot FROM admission WHERE admission_id = ?",
                (staged.admission_id,),
            )
            return Committed(int(found[0][0]), count, dropped, rewritten)

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
        family = "feature_set_" if table.startswith("feature_set_") else table
        key = DERIVED_KEYS.get(family)
        if key is None:
            raise DataError(
                f"{table} is not a derived table StorePort writes; it writes"
                f" {', '.join(DERIVED_KEYS)}",
                code="data_layer.derived_table_not_admitted",
            )
        target, run, part = f"lake.derived.{table}", self.lake.execute, DERIVED_PARTS[family]
        with self._writer:
            self.lake.register("deriving_rows", rows)
            self.lake.register("deriving_parts", parts)
            try:
                named = run(
                    "SELECT count(*), count(*) FILTER (WHERE n.recipe_key = ? AND EXISTS (SELECT 1"
                    f" FROM deriving_parts p WHERE p.part = {part} AND p.input_digest ="
                    " n.input_digest)) FROM deriving_rows n",
                    (recipe_key,),
                ).fetchall()[0]
                if named[0] != named[1]:
                    raise DataError(
                        f"{named[0] - named[1]} rows name another recipe, partition or inputs"
                        " than those derived; derive each recipe's partitions as they name",
                        code="data_layer.derived_rows_mixed",
                    )
                if table.startswith("feature_set_"):
                    self._feature_set(table, target)
                held = f"SELECT DISTINCT recipe_key, {part} AS part, input_digest FROM {target}"
                held += " WHERE input_digest IN (SELECT input_digest FROM deriving_parts)"
                fresh = f"SELECT n.* FROM deriving_rows n WHERE NOT EXISTS (SELECT 1 FROM ({held})"
                fresh += " h WHERE h.recipe_key = n.recipe_key AND h.input_digest = n.input_digest"
                fresh += f" AND h.part = {part})"
                count = run(f"SELECT count(*) FROM ({fresh})").fetchall()[0][0]
                if count:
                    run("BEGIN").complete()
                    try:
                        info = {"table": table, "recipe_key": recipe_key, "rows": count}
                        info |= {"parts": len(parts)}
                        run(
                            "CALL lake.set_commit_message('host', 'derived', extra_info => ?)",
                            (canonical(info).decode(),),
                        ).complete()
                        run(
                            f"INSERT INTO {target} BY NAME {fresh} ORDER BY {', '.join(key)}"
                        ).complete()
                        run("COMMIT").complete()
                    except BaseException:
                        run("ROLLBACK").complete()
                        raise
                filled = run(
                    f"SELECT {part}, input_digest, count(*) FROM deriving_rows GROUP BY ALL"
                ).fetchall()
            finally:
                self.lake.unregister("deriving_rows")
                self.lake.unregister("deriving_parts")
            share = int(getattr(rows, "nbytes", 0)) / max(len(rows), 1)
            self.log.cached(
                table,
                recipe_key,
                [(name, digest, held, round(share * held)) for name, digest, held in filled],
                compute_ms=compute_ms,
            )
            if rooted_by is not None:
                self.log.rooted(rooted_by, table, [(recipe_key, n, d) for n, d, _ in filled])
                self.log.flush()
        return count

    def collect(self, table: str) -> int:
        """Drop a derived table's partitions whose cache entries no workspace roots (the log's
        cache_root), as one catalog snapshot, and those entries: a cache holds no fact, so a
        partition no workspace reads goes. The rows dropped."""
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

    def _feature_set(self, table: str, target: str) -> None:
        """A feature set's wide table as catalog.sql declares it, made on its first partition and
        given a column for each one the rows bring that it does not hold yet, typed as the rows
        type it: a DOUBLE per factor, and the lineage the build stamps on each row. Every change
        is one catalog commit, so the table gains one schema version, never one per column: a
        read scans each schema version's inlined rows, and seventy versions cost a reader's first
        query seconds."""
        run = self.lake.execute
        made = run(
            "SELECT 1 FROM duckdb_tables() WHERE database_name = 'lake' AND schema_name = 'derived'"
            " AND table_name = ?",
            (table,),
        ).fetchall()
        held = {row[0] for row in run(f"DESCRIBE {target}").fetchall()} if made else set()
        added = [
            (column, kind)
            for column, kind in run(
                "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM deriving_rows)"
            ).fetchall()
            if column not in held
        ]
        if made and not added:
            return
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

    def maintain(self, now: datetime | None = None) -> dict[str, int]:
        """Keep the catalog compact and its bytes bounded, in one writer hold: flush inlined
        rows, rewrite files past their delete threshold (DERIVED_REWRITE in the derived schema),
        merge adjacent files, expire the snapshots older than KEEP_SNAPSHOTS, and delete the files
        no kept snapshot names once they have waited SETTLE_FILES. Expiry never reaches the oldest
        snapshot whose owed event the log lacks, which the next start's repair reads; when the log
        cannot be read, nothing expires. A retained backup holds its own copy of every file
        (backup.py, its files pool); a running one that finds a listed file deleted is refused by
        name (backup_unreadable) and is run again. The snapshots expired and the files deleted."""
        now = now or datetime.now(UTC)
        run = self.lake.execute
        with self._writer:
            run(
                "CALL lake.set_option('rewrite_delete_threshold', ?, schema => 'derived')",
                (DERIVED_REWRITE,),
            ).fetchall()
            for step in ("flush_inlined_data", "rewrite_data_files", "merge_adjacent_files"):
                run(f"CALL ducklake_{step}('lake')").fetchall()
            snapshots = run(
                "SELECT snapshot_id, snapshot_time, CASE WHEN commit_message = 'admission'"
                " THEN json_extract_string(commit_extra_info, '$.event_id') END"
                " FROM lake.snapshots()"
            ).fetchall()
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
            deleted = run(
                "CALL ducklake_cleanup_old_files('lake', older_than => ?)", (now - SETTLE_FILES,)
            ).fetchall()
        return {"expired": len(expiring), "deleted": len(deleted)}

    def close(self) -> None:
        """Close the Host's DuckDB session; the log is the caller's."""
        self.lake.close()

    def _registered(self, staged: Staged) -> None:
        """Refuse an admission naming a lineage the log does not register in its family."""
        parquet = self.root / "staging" / staged.admission_id / f"{staged.table}.parquet"
        columns = {
            row[0]
            for row in self.lake.execute(f"DESCRIBE SELECT * FROM '{sql_path(parquet)}'").fetchall()
        }
        named = set() if staged.provider_id is None else {("provider", staged.provider_id)}
        for column, family in LINEAGES.items():
            if column in columns and column != "provider_id":
                found = self.lake.execute(f"SELECT DISTINCT {column} FROM '{sql_path(parquet)}'")
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

    def _catalog_commit(
        self, staged: Staged, known: int, lag: int | None
    ) -> tuple[int, dict[str, int], int]:
        """One snapshot: the supersession of each world version the rows replace, then the rows and
        the later writes of held values, carrying the event it owes; the rows admitted, those
        dropped and the rewrites recorded, with no snapshot when it admits and records none. A
        batch whose changed rows were all dropped is recorded with no snapshot (`_judged`)."""
        table, admission = staged.table, staged.admission_id
        parquet = self.root / "staging" / admission / f"{table}.parquet"
        source = f"read_parquet('{sql_path(parquet)}')"
        world = table in WORLD_KEYS
        key = WORLD_KEYS[table] if world else RESULTS_KEYS[table]
        same = " AND ".join(f"t.{k} = n.{k}" for k in key)
        described = self.lake.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        columns = {row[0] for row in described}
        target = f"lake.{'world' if world else 'results'}.{table}"
        run = self.lake.execute
        rows, dropped, rewritten = f"SELECT * FROM {source}", {}, 0
        if world:  # read once, before the supersession changes which version is a key's newest
            rule = None if staged.lag is None or lag is None else (*staged.lag, lag)
            known_rows = self._known(table, source, columns, known, rule)
            run(f"CREATE OR REPLACE TEMP TABLE admitting AS {known_rows}").complete()
            fates = dict(run("SELECT fate, count(*) FROM temp.admitting GROUP BY ALL").fetchall())
            dropped = {fate: int(count) for fate, count in fates.items() if fate in DROPPED}
            rewritten = int(fates.get(REWRITTEN, 0))
            rows = "SELECT * EXCLUDE (fate, _held) FROM temp.admitting WHERE fate = 'admit'"
        try:
            parts, count = self._parts(table, rows) if world else (None, staged.row_count)
            if world and (count or staged.lag is not None):
                self._in_order(table, target, same, staged.lag)
            if not count and not rewritten:
                if dropped:
                    self._judged(staged, known, dropped)
                return 0, dropped, 0
            if not world and run(f"SELECT 1 FROM {target} t, ({rows}) n WHERE {same}").fetchone():
                raise DataError(
                    f"{table} already holds rows of these keys; an artifact's rows are admitted"
                    " once",
                    code="data_layer.result_key_present",
                )
            owed = self._owed(staged, known, parts, count, dropped, rewritten)
            self._write(staged, target, rows, columns, owed)
        finally:
            if world:
                run("DROP TABLE IF EXISTS temp.admitting").complete()
        return count, dropped, rewritten

    def _write(
        self, staged: Staged, target: str, rows: str, columns: set[str], owed: dict[str, Any]
    ) -> None:
        """The one catalog transaction: the event it owes, the supersession of each world version
        the rows replace, and of an interval's open version by its closing one, the later writes of
        held values, then the rows in their key order."""
        table, admission = staged.table, staged.admission_id
        run = self.lake.execute
        run("BEGIN").complete()
        try:
            run(
                "CALL lake.set_commit_message('host', 'admission', extra_info => ?)",
                (canonical(owed).decode(),),
            ).complete()
            if table in WORLD_KEYS:
                table_id, key = TABLE_IDS[table], WORLD_KEYS[table]
                same = " AND ".join(f"t.{k} = n.{k}" for k in key)
                pair = " AND ".join(f"o.{k} = c.{k}" for k in key)
                run(  # once, by its earliest replacement: an interval's pair replaces it once
                    f"INSERT INTO lake.world.superseded SELECT '{table_id}', t.version_id,"
                    f" min(n.known_asof), '{admission}' FROM {target} t, ({rows}) n"
                    f" WHERE {same} AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s"
                    f" WHERE s.table_id = '{table_id}' AND s.version_id = t.version_id)"
                    " GROUP BY t.version_id"
                ).complete()
                run(
                    f"INSERT INTO lake.world.superseded SELECT '{table_id}', o.version_id,"
                    f" c.known_asof, '{admission}' FROM ({rows}) o, ({rows}) c"
                    f" WHERE {pair} AND o._opening AND NOT c._opening"
                ).complete()
                if table in TIMED:
                    run(
                        f"INSERT INTO lake.world.rewritten SELECT '{table_id}', _held,"
                        f" {OBSERVED}, '{admission}' FROM temp.admitting"
                        f" WHERE fate = '{REWRITTEN}'"
                    ).complete()
                rows = f"SELECT * EXCLUDE (_opening), '{admission}' AS admission_id FROM ({rows})"
            order = ", ".join(WORLD_KEYS.get(table) or RESULTS_KEYS[table])
            run(f"INSERT INTO {target} BY NAME {rows} ORDER BY {order}").complete()
            run("COMMIT").complete()
        except BaseException:
            run("ROLLBACK").complete()
            raise

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
        a retried commit of it finds its event. No catalog commit carries the event, which names
        the snapshot it was judged against."""
        if self.log.read(
            "SELECT 1 FROM event WHERE type = 'admission.committed' AND subject_id = ?",
            (staged.admission_id,),
        ):
            return
        owed = self._owed(staged, known, [], 0, dropped, 0)
        snapshot = self.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
        row = {column: owed[column] for column in ENVELOPE if column in owed and column != "seq"}
        attrs = json.loads(owed["attrs"]) | {"catalog_snapshot": snapshot[0]}
        row["attrs"] = canonical(attrs).decode()
        self.log.append(row)

    def _known(
        self,
        table: str,
        source: str,
        columns: set[str],
        known: int,
        rule: tuple[str, int, int] | None,
    ) -> str:
        """Every staged row, conformed to its table's catalog types, with its version_id (a UUIDv7
        the engine mints), its known_at, its known_asof and its fate, as a query: `admit`, `same`
        (its content equals its key's newest version's, column by column, so however a writer
        stages one fact it is one version), `held_opening` (the open version of an interval world
        already holds), or a DROPPED reason. A row may carry `retracted` to withdraw its key's
        fact; a column it leaves out is null, and `retracted` false.

        An observed row's times are both `known`. A legacy row, staged under a lag `rule` (its
        name, version and seconds), names the rule and version; its known_at is empty and its
        known_asof is its publication time, or its lag after the close of its market's first
        session on or after its valid date; the market is the row's own listing's, known only while
        that listing has had one market. An UNMARKETED row (a document) is known its lag after its
        own valid time, its availability. A legacy row of a key world already holds is a change
        observed now: known no earlier than its admission or the version it replaces, so no read
        before then sees it; but a legacy document over a legacy version (its known_at empty) is
        another 0.2.7 fact with its own availability, known then, never before the version it
        replaces. A legacy interval with an end, for a key world does not hold yet, is
        written as its two versions: the open one known from its start, the closing one from its
        end. In a TIMED table, a held key is superseded only by a row its source wrote after the
        newest version's value was last written (its ordering time, `newest`), and a later write
        of that same value is REWRITTEN."""
        at = f"make_timestamptz({known}::BIGINT)"
        key = WORLD_KEYS[table]
        declared = self._declared(table, columns)
        conformed = ", ".join(
            f"CAST({name} AS {kind}) AS {name}"
            if name in columns
            else f"{'false' if name == 'retracted' else 'NULL'}::{kind} AS {name}"
            for name, kind in declared.items()
        )
        rows = f"SELECT {conformed}, false AS _opening FROM {source}"
        interval = LEGACY_VALID[table] if rule is not None else ()
        if len(interval) == 2 and interval[0] in columns:  # an interval's open version, beside
            end = interval[0]  # its closing one
            rows = (
                f"SELECT * FROM ({rows}) UNION ALL SELECT * REPLACE (NULL::{declared[end]} AS"
                f" {end}, true AS _opening) FROM ({rows}) WHERE {end} IS NOT NULL"
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
            return f"SELECT n.*, {at} AS known_at, {at} AS known_asof, {fate} FROM ({rows}) n{held}"
        valid = ", ".join(f"n.{c}" for c in LEGACY_VALID[table] if c in columns)
        name, version, lag = rule
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
        if "published_at" in columns:
            derived = f"coalesce(n.published_at, {derived})"
        market, joined = "n.market", ""
        if table != "listing":
            market = "l.market"
            joined = (  # a listing that changed market has no market for a date before the change
                " LEFT JOIN (SELECT listing_id, any_value(market) AS market FROM lake.world.listing"
                " WHERE NOT retracted GROUP BY listing_id HAVING count(DISTINCT market) = 1) l"
                " ON l.listing_id = n.listing_id"
            )
        asof = (
            f"CASE WHEN p.version_id IS NULL THEN {derived}"
            f" ELSE greatest({derived}, {at}, p.known_asof) END"
        )
        return (
            f"SELECT n.*, NULL::TIMESTAMPTZ AS known_at, {asof} AS known_asof, {fate},"
            f" {named} FROM ({rows}) n{joined} ASOF LEFT JOIN {_current('session')} a"
            f" ON a.market = {market} AND coalesce({valid}) <= a.session_date{held}"
        )

    def _declared(self, table: str, columns: set[str]) -> dict[str, str]:
        """A world table's content and SOURCE columns with their catalog types, as the engine
        holds them; refused by name when the rows carry a column the table does not hold."""
        described = self.lake.execute(f"DESCRIBE lake.world.{table}").fetchall()
        declared = {row[0]: row[1] for row in described if row[0] not in ADMITTED}
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
        found = self.lake.execute(
            f"SELECT {part} AS part, min(epoch_us(known_asof)), max(epoch_us(known_asof)),"
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
        staged: Staged,
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
            "provider_id": staged.provider_id,
            "dataset": staged.table,
            "row_count": count,
            "known_us": known,
            "parts": whole if parts is None else parts,
        }
        if staged.lag is not None:
            attrs |= {"lag_rule": staged.lag[0], "lag_rule_version": staged.lag[1]}
        if staged.artifacts:
            attrs["artifact_ids"] = list(staged.artifacts)
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


@contextmanager
def stage(
    root: Path,
    table: str,
    provider_id: str | None,
    rows: Any,
    *,
    lag: tuple[str, int] | None = None,
    floor_bytes: int = FREE_FLOOR_BYTES,
) -> Iterator[Staged]:
    """Write one admission's rows, an Arrow table or record batch reader, under staging as they
    came, in Arrow from the caller to the Parquet; the commit request it yields
    is all the writer needs. `provider_id` is a world table's provider lineage in the log, and None
    for a results table, whose rows' artifact_id is their lineage. Its producer holds
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
    staged = admissible(table, provider_id, rows, legacy=lag is not None)
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
        (root / "staging" / admission).mkdir()
        pq.write_table(staged, root / "staging" / admission / f"{table}.parquet")
        artifacts = (
            tuple(sorted(set(staged.column("artifact_id").to_pylist())))
            if table in RESULTS_KEYS
            else ()
        )
        yield Staged(admission, table, provider_id, staged.num_rows, lag, artifacts)
    finally:
        drop(held, lock)


def newest(table: str) -> str:
    """Each key's newest version in a world table, every column, as a subquery. A TIMED table's
    also carries `ordered_at`: its value's last write by its source, the later of the version's own
    and each world.rewritten write of it, -infinity when there is none, so a row with a source time
    orders after a version without one."""
    table_id = TABLE_IDS[table]
    current = (
        f"SELECT x.* FROM lake.world.{table} x WHERE NOT EXISTS (SELECT 1 FROM"
        f" lake.world.superseded s WHERE s.table_id = '{table_id}'"
        " AND s.version_id = x.version_id)"
    )
    if table not in TIMED:
        return f"({current})"
    return (
        f"(SELECT c.*, coalesce(greatest(c.{OBSERVED}, w.latest), '-infinity'::TIMESTAMPTZ)"
        f" AS ordered_at FROM ({current}) c LEFT JOIN (SELECT version_id, max({OBSERVED})"
        f" AS latest FROM lake.world.rewritten WHERE table_id = '{table_id}' GROUP BY ALL) w"
        " ON w.version_id = c.version_id)"
    )


def feature_table(feature_set_id: str) -> str:
    """A feature set's table in `derived`, named by its lineage's id and never its name, so a
    renamed feature set keeps every cached partition: feature_set_ and the id's hex digits."""
    return "feature_set_" + feature_set_id.removeprefix("blake3:")


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


def repair(log: HostLog, lake: Any) -> list[str]:
    """Append each event a catalog snapshot carried and the log lacks; returns their ids.

    An owed event keeps every field its writer made, its id, version, workspace, actor, trace and
    time included; only its recording time is new, after the moment it carries, and its
    attributes name the snapshot. Running it again adds none. Only an admission's commit owes an
    event; a derived write's commit info only describes its rows.
    """
    carried = lake.execute(
        "SELECT snapshot_id, commit_extra_info FROM lake.snapshots()"
        " WHERE commit_message = 'admission' AND commit_extra_info IS NOT NULL"
        " ORDER BY snapshot_id"
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
    take it meanwhile, settles each array body a crash left staged under blobs/ (placed when its
    profile was committed, else collected; `arrays.settle`), and collects every file under data/
    the catalog does not name.
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


def _current(table: str) -> str:
    """A world table's current versions, as a subquery: those not withdrawn and not replaced."""
    return (
        f"(SELECT * FROM lake.world.{table} x WHERE NOT x.retracted AND NOT EXISTS (SELECT 1"
        f" FROM lake.world.superseded s WHERE s.table_id = '{TABLE_IDS[table]}'"
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
