"""A workspace's market tables admitted into world once per machine, through the Host's StorePort.

0.2.7 keeps a workspace's listings, provider symbols, raw daily bars, the provider's adjusted
closes and corporate actions in its market-data.duckdb. When the Host opens the workspace, `admit`
attaches that database read-only to the Host's catalog session and copies its rows into the world
tables, in order:

1. the sessions of each listing's market, from its market profile's exchange calendars, observed;
2. the listings, each named by the uuid5 of its 0.2.7 id, on the market its profile's calendar
   names (`market_of`), with 0.2.7's universe class (its `mic`) as data, and their provider
   symbols, each listing with two aliases: its 0.2.7 id (LISTING_ALIAS) and its display symbol
   (DISPLAY_ALIAS);
3. the raw bars with the provider's adjusted close, then the corporate actions, each with its
   source's own last write (`source_observed_at`);
4. each workspace's original bar/action revision as its registered log fact, retaining its
   original key, fields and observation time for that workspace's audit cutoff read;

all but the sessions as legacy batches under the market lag rule (MARKET_CLOSE). Only the bars and
actions whose values differ from world's newest version of their key become rows; StorePort admits
a changed row of a held key only when its source wrote it later, known from its admission.

Each admission records the tables' marks in its `admission` event: each table's row count and the
exclusive-or of its rows' hashes, a content fingerprint. The next open reads the marks first;
equal marks end it there.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from itertools import groupby
from pathlib import Path
from typing import Any, Final

from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    MembershipEpoch,
    MembershipEvent,
    MembershipSchedule,
    UniverseBootstrapRecord,
    resolve_membership_schedule,
)
from alphalattice.foundation.market_data_ops.storage.universe_workspace import (
    UniverseWorkspaceReader,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.log import attach
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.root import sql_path
from alphalattice.kernel.data_layer.store import as_columns, newest, stage
from alphalattice.kernel.quant.sector_history import SectorReclassification

DATASET: Final = "market"
MARKET_CLOSE: Final = ("market-close", 1)
"""The lag rule of 0.2.7's market rows, which carry no knowledge time: known at their session's
close (lag_seconds 0)."""
LISTING_ALIAS: Final = "alphalattice-0.2.7-listing"
DISPLAY_ALIAS: Final = "display"
CALENDARS: Final = "exchange-calendars"
"""The provider of the sessions: the exchange_calendars package the market profile names."""
PROVIDERS: Final = {
    LISTING_ALIAS: "A listing's id in 0.2.7, kept as an alias of its world listing.",
    DISPLAY_ALIAS: "A listing's display symbol, kept as an alias of its world listing.",
    CALENDARS: "The exchange_calendars package: each market's sessions, opens and closes.",
}
LISTINGS: Final = uuid.uuid5(uuid.NAMESPACE_URL, "alphalattice:world:listing")
"""The namespace of world listing ids: a 0.2.7 listing's world id is uuid5 of its id in it."""
KINDS: Final = {
    "SPLIT": "split",
    "CASH_DIVIDEND": "dividend",
    "CAPITAL_GAIN": "dividend",
    "SPIN_OFF": "spinoff",
}
"""0.2.7's action kinds in world's vocabulary; the action's id keeps 0.2.7's kind, so a capital
gain, a cash distribution, stays apart from a dividend on its day."""
FULL_DAY: Final = timedelta(hours=6, minutes=30)
CALENDAR_REACH: Final = timedelta(days=10)
"""Sessions read past the last valid date, so a date after the last session still finds its next."""
REVISIONS: Final = {
    "bar_revision": (
        "market.bar_revision",
        (
            "revision_id",
            "listing_id",
            "provider",
            "session_date",
            "prior_payload_hash",
            "next_payload_hash",
            "observed_at",
            "reason",
        ),
    ),
    "corporate_action_revision": (
        "market.action_revision",
        (
            "revision_id",
            "listing_id",
            "provider",
            "effective_date",
            "action_kind",
            "revision_kind",
            "prior_payload_hash",
            "next_payload_hash",
            "observed_at",
        ),
    ),
}
"""The original revision facts, by source table: their registered event type and exact columns.
Earlier values kept only as digests remain digests; no numerical history is inferred."""
TABLES: Final = (
    "listing",
    "provider_symbol_mapping",
    "market_profile",
    "raw_daily_bar_current",
    "provider_adjusted_close_current",
    "corporate_action_current",
    "universe_manifest",
    "universe_manifest_listing",
    "universe_bootstrap",
    "universe_membership_event",
    *REVISIONS,
)
"""The 0.2.7 tables the admission reads. Each one's mark is its row count and the exclusive-or of
its rows' hashes: a content fingerprint, which no write time can leave unchanged."""
WORKSPACE: Final = "market_workspace"
"""The name the workspace's database is attached under, read-only, for one admission."""
DROPS: Final = (
    "DROP TABLE IF EXISTS temp.market_ids",
    "DROP TABLE IF EXISTS temp.market_providers",
    "DROP TABLE IF EXISTS temp.market_bars",
)
"""The admission's temporary tables, dropped when it ends however it ends."""
ADMISSION_ROWS: Final = 1 << 21
"""The most bars one admission stages, in whole years: a market's history in one commit where it
fits, which DuckDB judges and writes across its threads at once (ls1-daily's 1,252,430 bars in
1.4 s against 2.9 s as eleven years); a longer one in groups of years under it, so its Arrow
batch stays bounded."""
_WRITTEN: Final = "strftime({0}, '%Y-%m-%d %H:%M:%S.%f+00')"
"""A 0.2.7 write time, which 0.2.7 keeps in UTC, as StorePort reads a staged time."""


def table_marks(run: Any, schema: str = "") -> dict[str, list[int]]:
    """Each of TABLES' marks, read with `run` from the database `schema` names (with its dot)."""
    return {
        table: run(f"SELECT [count(*), bit_xor(hash(t))] FROM {schema}{table} t").fetchone()[0]
        for table in TABLES
    }


UNIVERSE_EVENTS: Final = {
    "ENTRY": "universe.member_added",
    "EXIT": "universe.member_removed",
}
"""The journal's two membership changes, registered with the machine log."""


def _universe_journal(events: Sequence[Any], workspace_id: str) -> None:
    """Import 0.2.7 journal facts once, keeping the original content digest and knowledge time."""
    for kind in UNIVERSE_EVENTS.values():
        history.define(
            "event_type",
            kind,
            "A universe membership changed on its effective session.",
            {
                "work": False,
                "subject": "universe",
                "required": ["content_digest"],
                "text": "none",
                "durable": True,
            },
        )
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for event in events:
        attrs = {
            "sequence": event.sequence,
            "listing_id": event.listing_id,
            "effective_session": event.effective_session.isoformat(),
            "observed_at": event.observed_at.isoformat(),
            "authority": event.authority,
            "reference_hash": event.reference_hash,
            "manifest_revision": event.manifest_revision,
            "content_digest": event.event_hash,
        }
        history.record_new(
            "SELECT 1 FROM event WHERE subject_kind = 'universe' AND subject_id = ?"
            " AND type IN ('universe.member_added', 'universe.member_removed')"
            " AND json_extract(attrs, '$.content_digest') = ?",
            (universe_lineage(workspace_id, event.market_profile_id), event.event_hash),
            UNIVERSE_EVENTS[event.kind],
            ("universe", universe_lineage(workspace_id, event.market_profile_id)),
            attrs,
            occurred_us=(event.decided_at - epoch) // timedelta(microseconds=1),
            workspace=None,
            node="market_data_ops",
        )


def membership_events(
    market_profile_id: str,
    cutoff_us: int | None = None,
    *,
    workspace_id: str,
    connection: Any = None,
) -> tuple[Any, ...]:
    """Rebuild the 0.2.7 journal from the machine's registered events."""
    connection = _universe_surface(connection)
    rows = _log_rows(
        "SELECT type, occurred_us, attrs FROM event WHERE subject_kind = 'universe'"
        " AND subject_id = ? AND type IN ('universe.member_added', 'universe.member_removed')"
        " AND (? IS NULL OR occurred_us <= ?)"
        " ORDER BY CAST(json_extract(attrs, '$.sequence') AS INTEGER)",
        (universe_lineage(workspace_id, market_profile_id), cutoff_us, cutoff_us),
    )
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    values = []
    for kind, occurred, raw in rows:
        attrs = json.loads(raw)
        values.append(
            MembershipEvent(
                market_profile_id=market_profile_id,
                sequence=attrs["sequence"],
                listing_id=attrs["listing_id"],
                kind="ENTRY" if kind == UNIVERSE_EVENTS["ENTRY"] else "EXIT",
                effective_session=date.fromisoformat(attrs["effective_session"]),
                observed_at=datetime.fromisoformat(attrs["observed_at"]),
                decided_at=epoch + timedelta(microseconds=occurred),
                authority=attrs["authority"],
                reference_hash=attrs["reference_hash"],
                manifest_revision=attrs["manifest_revision"],
                event_hash=attrs["content_digest"],
            )
        )
    return tuple(values)


def world_listing(listing_id: str) -> str:
    """A 0.2.7 listing's world id."""
    return str(uuid.uuid5(LISTINGS, listing_id))


def universe_name(workspace_id: str, profile_id: str) -> str:
    """A workspace's governed profile, named by their registered ids."""
    return f"{workspace_id}/{profile_id}"


def universe_lineage(workspace_id: str, profile_id: str) -> str:
    """The universe definition id assigned to a workspace's governed profile."""
    return definition_id("universe", universe_name(workspace_id, profile_id))


def universe_set_digest(listing_ids: Iterable[str]) -> str:
    """Name a frozen member set by the canonical sorted world listing IDs."""
    return digest(canonical(sorted({world_listing(listing) for listing in listing_ids})))


def universe_set_id(listing_ids: Iterable[str]) -> str:
    """The registered universe lineage of a frozen member set."""
    return definition_id("universe", universe_set_digest(listing_ids))


def admit_member_set(listing_ids: Iterable[str], valid_from: date, provider: str) -> str:
    """Admit a frozen set through StorePort, with the source's registered legacy lag."""
    listings = sorted(set(listing_ids))
    content = universe_set_digest(listings)
    identity = definition_id("universe", content)
    history.define(
        "universe", content, "A frozen universe member set.", {"content_digest": content}
    )
    _commit_universe_members(identity, ((item, valid_from, None) for item in listings), provider)
    return identity


def _commit_universe_members(
    identity: str, intervals: Iterable[tuple[str, date, date | None]], provider: str
) -> None:
    """Admit one universe's legacy intervals through its registered StorePort."""
    port = history.store()
    if port is None:
        raise DataError(
            "the Host's universe store is not open", code="data_layer.history_unavailable"
        )
    rows = [
        {
            "universe_id": identity,
            "listing_id": world_listing(listing),
            "valid_from": start.isoformat(),
            "valid_to": None if end is None else end.isoformat(),
        }
        for listing, start, end in intervals
    ]
    if rows:
        with stage(
            port.root,
            "universe_member",
            definition_id("provider", provider),
            _arrow(rows),
            lag=MARKET_CLOSE,
        ) as staged:
            port.commit(staged)


def _log_rows(sql: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Rows of `sql` over the log: in the process that holds it, through the HostLog's own
    connection, its open batch included; in any other, through a read-only one (`log.attach`).
    Both are Python's sqlite3, the one SQLite library a process opens the log with. No rows in a
    data root with no log yet."""
    host = history.opened_log()
    held = host[0] if host is not None else history.holding()
    if held is not None:
        return [tuple(row) for row in held.read(sql, tuple(params))]
    root = history.data_root()
    if not (root / "machine" / "log.sqlite").is_file():
        return []
    with closing(attach(root)) as con:
        return [tuple(row) for row in con.execute(sql, tuple(params)).fetchall()]


def _universe_surface(connection: Any = None) -> Any:
    """Use the admitted reader's snapshot, or the open Host's surface for direct reads."""
    if connection is not None:
        return connection
    port = history.store()
    if port is None:
        raise DataError(
            "the Host's universe store is not open", code="data_layer.history_unavailable"
        )
    port.log.flush()
    return port.lake


def _member_ids(connection: Any, set_id: str, cutoff_us: int) -> tuple[str, ...]:
    """Read a content-keyed set's 0.2.7 listing aliases and verify its world content."""
    rows = connection.execute(
        "SELECT DISTINCT CAST(m.listing_id AS VARCHAR), p.symbol FROM universe_member_asof(?) m"
        " JOIN lake.world.provider_symbol p USING (listing_id) WHERE m.universe_id = ?"
        " AND p.provider_id = ? AND NOT p.retracted AND p.known_asof <= make_timestamptz(?)"
        " AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s WHERE s.table_id = ?"
        " AND s.version_id = p.version_id"
        " AND s.superseded_asof <= make_timestamptz(?)) ORDER BY ALL",
        [
            cutoff_us,
            set_id,
            definition_id("provider", LISTING_ALIAS),
            cutoff_us,
            definition_id("world_table", "provider_symbol"),
            cutoff_us,
        ],
    ).fetchall()
    definition = _log_rows(
        "SELECT spec FROM definition_version WHERE definition_id = ? AND version = 1",
        (set_id,),
    )
    if (
        len(definition) != 1
        or digest(canonical(sorted({row[0] for row in rows})))
        != json.loads(definition[0][0])["content_digest"]
    ):
        raise ValueError("market_data_ops.universe_manifest_membership_changed")
    return tuple(sorted(row[1] for row in rows))


def admit_universe_bootstrap(record: Any, provider: str, workspace_id: str) -> None:
    """Record one frozen bootstrap, with its cohort held as a world member set."""
    cohort = admit_member_set(record.cohort_listing_ids, record.history_start, provider)
    history.define(
        "event_type",
        "universe.bootstrapped",
        "A universe's initial cohort and boundary were frozen.",
        {
            "work": False,
            "subject": "universe",
            "required": ["member_set", "record_hash"],
            "text": "none",
            "durable": True,
        },
    )
    attrs = asdict(record)
    attrs.pop("cohort_listing_ids")
    attrs["member_set"] = cohort
    for key in ("t0_session", "history_start", "source_observed_at", "admitted_at"):
        attrs[key] = attrs[key].isoformat()
    opened = history.opened_log()
    if opened is None:
        raise DataError(
            "the Host's universe log is not open", code="data_layer.history_unavailable"
        )
    log = opened[0]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    with log.unit() as con:
        held = con.execute(
            "SELECT attrs FROM event WHERE type = 'universe.bootstrapped' AND subject_id = ?",
            (universe_lineage(workspace_id, record.market_profile_id),),
        ).fetchall()
        if held:
            if len(held) != 1 or json.loads(held[0][0]) != attrs:
                raise ValueError("universe bootstrap is already recorded with different content")
            return
        log.record(
            "universe.bootstrapped",
            ("universe", universe_lineage(workspace_id, record.market_profile_id)),
            attrs,
            occurred_us=(record.admitted_at - epoch) // timedelta(microseconds=1),
            node="market_data_ops",
        )


def universe_bootstrap(
    market_profile_id: str, cutoff_us: int, *, workspace_id: str, connection: Any = None
) -> Any:
    """Rebuild the frozen bootstrap boundary and its cohort at the selection's cutoff."""
    connection = _universe_surface(connection)
    found = _log_rows(
        "SELECT attrs FROM event WHERE type = 'universe.bootstrapped'"
        " AND subject_id = ? AND occurred_us <= ?",
        (universe_lineage(workspace_id, market_profile_id), cutoff_us),
    )
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("universe bootstrap is ambiguous")
    attrs = json.loads(found[0][0])
    set_id = attrs.pop("member_set")
    attrs["cohort_listing_ids"] = _member_ids(connection, set_id, cutoff_us)
    for key in ("t0_session", "history_start"):
        attrs[key] = date.fromisoformat(attrs[key])
    for key in ("source_observed_at", "admitted_at"):
        attrs[key] = datetime.fromisoformat(attrs[key])
    return UniverseBootstrapRecord(**attrs)


def admit_universe_manifest(manifest: Any, workspace_id: str) -> str:
    """Register an immutable manifest version and its content-keyed world member set."""
    listing_ids = [item.listing_id for item in manifest.listings]
    set_id = admit_member_set(
        listing_ids, manifest.profile.manifest_as_of, manifest.profile.provider
    )
    spec = asdict(manifest)
    spec["profile"]["manifest_as_of"] = manifest.profile.manifest_as_of.isoformat()
    spec["member_set"] = set_id
    spec = json.loads(json.dumps(spec))  # the version's canonical JSON shape, including tuples
    opened = history.opened_log()
    if opened is None:
        raise DataError(
            "the Host's universe log is not open", code="data_layer.history_unavailable"
        )
    log = opened[0]
    name = universe_name(workspace_id, manifest.profile.market_profile_id)
    with log.unit() as con:
        versions = con.execute(
            "SELECT version, spec FROM definition WHERE family = 'universe'"
            " AND name = ? ORDER BY version",
            (name,),
        ).fetchall()
        for _, raw in versions:
            previous = json.loads(raw)
            if previous.get("revision_sha256") == manifest.revision_sha256:
                if previous != spec:
                    raise ValueError(
                        "universe manifest revision disagrees with its registered content"
                    )
                return set_id
        version = len(versions) + 1
        key = {"family": "universe", "name": name, "version": version}
        body = key | {
            "status": "standard",
            "spec": spec,
            "owner": "market_data_ops",
            "text": "A frozen market universe manifest.",
        }
        if version > 1:
            body["relation"] = "not_comparable"
        log.record(
            "definition.registered",
            ("definition", f"universe/{name}"),
            key,
            text=canonical(body).decode(),
            node="market_data_ops",
        )
    return set_id


def load_universe_manifest(
    manifest_id: str, cutoff_us: int, *, workspace_id: str, connection: Any = None
) -> Any:
    """Read one immutable manifest and its member set at the selection's cutoff."""
    connection = _universe_surface(connection)
    found = _log_rows(
        "SELECT v.spec FROM definition_version v JOIN definition_lineage l"
        " USING (definition_id)"
        " WHERE l.family = 'universe' AND instr(l.name, ?) = 1"
        " AND json_extract(v.spec, '$.manifest_id') = ?",
        (workspace_id + "/", manifest_id),
    )
    if len(found) != 1:
        raise ValueError("workspace manifest does not exist or is ambiguous")
    spec = json.loads(found[0][0])
    universe = spec.pop("member_set")
    listings = tuple(ManifestListing(**item) for item in spec.pop("listings"))
    expected = tuple(sorted(item.listing_id for item in listings))
    if (
        universe_set_id(expected) != universe
        or _member_ids(connection, universe, cutoff_us) != expected
    ):
        raise ValueError("market_data_ops.universe_manifest_membership_changed")
    profile = spec.pop("profile")
    profile["manifest_as_of"] = date.fromisoformat(profile["manifest_as_of"])
    spec["profile"] = MarketProfile(**profile)
    spec["listings"] = listings
    for key in ("universe_components", "qualification_obligations"):
        spec[key] = tuple(spec[key])
    return UniverseManifest(**spec)


def membership_schedule(
    market_profile_id: str,
    sessions: Sequence[date],
    fallback_listing_ids: Sequence[str],
    cutoff_us: int,
    *,
    workspace_id: str,
    connection: Any = None,
) -> Any:
    """Read each session's membership at its close or the selection's earlier cutoff."""
    connection = _universe_surface(connection)
    bootstrap = universe_bootstrap(
        market_profile_id, cutoff_us, workspace_id=workspace_id, connection=connection
    )
    events = membership_events(
        market_profile_id, cutoff_us, workspace_id=workspace_id, connection=connection
    )
    reference = resolve_membership_schedule(
        sessions=sessions,
        bootstrap=bootstrap,
        events=events,
        fallback_listing_ids=fallback_listing_ids,
    )
    if bootstrap is None:
        return reference
    profile = _log_rows(
        "SELECT v.spec FROM definition_version v JOIN definition_name n"
        " USING (definition_id)"
        " WHERE n.family = 'universe' AND n.name = ? ORDER BY v.version DESC LIMIT 1",
        (universe_name(workspace_id, market_profile_id),),
    )
    if not profile:
        raise ValueError("universe manifest is not registered")
    market = market_of(json.loads(profile[0][0])["profile"]["calendar_id"])
    days = list(reference.sessions)
    rows = connection.execute(
        "WITH days AS (SELECT unnest(?::DATE[]) AS day),"
        " clocks AS (SELECT session_date, max(close_utc) AS close_utc FROM lake.world.session"
        " WHERE market = ? AND NOT retracted GROUP BY session_date),"
        " aliases AS (SELECT p.*, row_number() OVER (PARTITION BY p.listing_id"
        " ORDER BY p.valid_from DESC) AS n"
        " FROM lake.world.provider_symbol p WHERE p.provider_id = ? AND NOT p.retracted"
        " AND p.known_asof <= make_timestamptz(?) AND NOT EXISTS"
        " (SELECT 1 FROM lake.world.superseded s WHERE s.table_id = ?"
        " AND s.version_id = p.version_id AND s.superseded_asof <= make_timestamptz(?)))"
        " SELECT days.day, max(clocks.close_utc),"
        " list(a.symbol ORDER BY a.symbol) FILTER (WHERE a.symbol IS NOT NULL)"
        " FROM days LEFT JOIN clocks ON clocks.session_date = days.day"
        " LEFT JOIN universe_member_versions m ON m.universe_id = ? AND NOT m.retracted"
        " AND m.valid_from <= days.day AND (m.valid_to IS NULL OR days.day < m.valid_to)"
        " AND m.known_asof <= least(clocks.close_utc, make_timestamptz(?))"
        " AND (m.visible_to IS NULL OR m.visible_to > least(clocks.close_utc, make_timestamptz(?)))"
        " LEFT JOIN aliases a ON a.listing_id = m.listing_id AND a.n = 1"
        " GROUP BY days.day ORDER BY days.day",
        [
            days,
            market,
            definition_id("provider", LISTING_ALIAS),
            cutoff_us,
            definition_id("world_table", "provider_symbol"),
            cutoff_us,
            universe_lineage(workspace_id, market_profile_id),
            cutoff_us,
            cutoff_us,
        ],
    ).fetchall()
    if any(clock is None for _, clock, _ in rows):
        raise ValueError("universe membership session close is unavailable")
    # The imported interval's legacy lag is its effective close. The journal is the owner of
    # each decision time, including decisions after that close. Apply its facts only once known,
    # so the interval projection cannot move a historical selection before the decision.
    journal = membership_events(market_profile_id, workspace_id=workspace_id, connection=connection)
    by_listing: dict[str, list[Any]] = {}
    for event in sorted(journal, key=lambda item: (item.effective_session, item.sequence)):
        by_listing.setdefault(event.listing_id, []).append(event)
    cohort = set(bootstrap.cohort_listing_ids)
    selected_at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=cutoff_us)
    members = []
    for day, close, values in rows:
        held = set(values or ())
        known = min(close, selected_at)
        for listing, changes in by_listing.items():
            last = next(
                (
                    event
                    for event in reversed(changes)
                    if event.effective_session <= day
                    and (event.effective_session < day or close <= selected_at)
                    and event.decided_at <= known
                ),
                None,
            )
            if (last.kind == "ENTRY") if last is not None else (listing in cohort):
                held.add(listing)
            else:
                held.discard(listing)
        members.append((day, tuple(sorted(held))))
    epochs = []
    for _, group in groupby(members, key=lambda item: item[1]):
        part = list(group)
        epochs.append(MembershipEpoch(part[0][0], part[-1][0], part[0][1]))
    return MembershipSchedule(
        reference.sessions,
        tuple(epochs),
        reference.basis_ranges,
        bootstrap,
        reference.journal_sequence,
        reference.admitted_listing_ids,
    )


def load_universe_manifest_revision(
    manifest_revision: str, cutoff_us: int, *, workspace_id: str, connection: Any = None
) -> Any:
    """Resolve an immutable manifest revision from the machine's definitions."""
    connection = _universe_surface(connection)
    rows = _log_rows(
        "SELECT json_extract(v.spec, '$.manifest_id') FROM definition_version v"
        " JOIN definition_lineage l USING (definition_id)"
        " WHERE l.family = 'universe' AND instr(l.name, ?) = 1"
        " AND json_extract(v.spec, '$.revision_sha256') = ?",
        (workspace_id + "/", manifest_revision),
    )
    if len(rows) != 1:
        raise ValueError("workspace manifest revision is missing or ambiguous")
    return load_universe_manifest(
        str(rows[0][0]), cutoff_us, workspace_id=workspace_id, connection=connection
    )


def _admit_universe_intervals(
    bootstrap: Any, events: Sequence[Any], provider: str, workspace_id: str
) -> None:
    """Admit the bootstrap cohort and its journal as legacy membership intervals."""
    identity = history.definition_id(
        "universe", universe_name(workspace_id, bootstrap.market_profile_id)
    )
    if identity is None:
        raise ValueError("universe manifest is not registered")
    active = {listing: bootstrap.history_start for listing in bootstrap.cohort_listing_ids}
    intervals: list[tuple[str, date, date | None]] = []
    for event in sorted(events, key=lambda item: (item.effective_session, item.sequence)):
        if event.kind == "ENTRY":
            if event.listing_id in active:
                raise ValueError("membership entry for a listing already present")
            active[event.listing_id] = event.effective_session
        else:
            if event.listing_id not in active:
                raise ValueError("membership exit for a listing not present")
            start = active.pop(event.listing_id)
            if start < event.effective_session:
                intervals.append((event.listing_id, start, event.effective_session))
    intervals.extend((listing, start, None) for listing, start in active.items())
    _commit_universe_members(identity, intervals, provider)


def admit_universe(database: Path, workspace_id: str) -> None:
    """Admit market and universe data together through their shared admission edge."""
    admit(database, workspace_id)


def _admit_universe_contents(database: Path, workspace_id: str, attached: Any) -> None:
    """Copy a workspace's manifests, bootstrap and journal before its admission is recorded,
    read through `attached`, a cursor of the admission's attachment of the file, so the
    admission opens the file no second time."""
    connection = attached.cursor()
    connection.execute(f"USE {WORKSPACE}")
    market = UniverseWorkspaceReader(database.parent, through=connection)
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "universe_manifest" not in tables:
            return
        manifest_ids = [
            row[0]
            for row in connection.execute(
                "SELECT manifest_id FROM universe_manifest"
                " ORDER BY market_profile_id, as_of_date, manifest_id"
            ).fetchall()
        ]
        profiles = (
            [
                row[0]
                for row in connection.execute(
                    "SELECT market_profile_id FROM universe_bootstrap ORDER BY market_profile_id"
                ).fetchall()
            ]
            if "universe_bootstrap" in tables
            else []
        )
        providers: dict[str, str] = {}
        for name in manifest_ids:
            manifest = market._workspace_load_universe_manifest(name)
            admit_universe_manifest(manifest, workspace_id)
            providers[manifest.profile.market_profile_id] = manifest.profile.provider
        for profile in profiles:
            bootstrap = market._workspace_universe_bootstrap(profile)
            if bootstrap is None:
                continue
            events = market._workspace_membership_events(profile)
            _universe_journal(events, workspace_id)
            admit_universe_bootstrap(bootstrap, providers[profile], workspace_id)
            _admit_universe_intervals(bootstrap, events, providers[profile], workspace_id)
    finally:
        connection.close()


def market_of(calendar: str) -> str:
    """A market profile calendar's market: the ISO 10383 MIC its id names first, so XNYS_XNAS (or
    XNAS+XNYS, as older profiles join them) is XNYS (XNAS)."""
    return re.split("[_+]", calendar)[0]


def admit(database: Path, workspace_id: str) -> dict[str, Any] | None:
    """Admit the workspace's market tables in `database` into world; the rows each world table took,
    those StorePort dropped and the later writes of held values it recorded, or None when nothing
    was read: no Host history, no market database, or marks equal to the last admission's. Refused
    by name: a market whose calendars disagree, and as StorePort refuses."""
    port = history.store()
    if port is None or not database.is_file():
        return None
    run = port.lake.execute
    run(f"ATTACH '{sql_path(database)}' AS {WORKSPACE} (READ_ONLY)").complete()
    try:
        held = run(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = ?", [WORKSPACE]
        ).fetchall()
        if not {row[0] for row in held} >= set(TABLES):
            return None
        marks = table_marks(run, f"{WORKSPACE}.")
        last = port.log.read(
            "SELECT json_extract(attrs, '$.marks') FROM event WHERE type = 'admission'"
            " AND subject_id = ? AND json_extract(attrs, '$.dataset') = ? ORDER BY seq DESC"
            " LIMIT 1",
            (workspace_id, DATASET),
        )
        if last and json.loads(last[0][0]) == marks:
            return None
        started = port.log.clock()
        providers = _register(run)
        admitted = _admit(port, providers)
        _admit_universe_contents(database, workspace_id, port.lake)
        _revisions(port, workspace_id)
        history.record(
            "admission",
            ("workspace", workspace_id),
            {"provider": sorted(providers), "dataset": DATASET, "marks": marks} | admitted,
            phase="completed",
            started_us=started,
            node="market_data_ops",
        )
        history.flush()
        return admitted
    finally:
        for drop in DROPS:
            run(drop).complete()
        run(f"DETACH {WORKSPACE}").complete()


def sector_marks(database: Path) -> dict[str, Any]:
    """The sector tables' content fingerprints, as recorded by their admission."""
    connection = open_workspace_database(database, read_only=True)
    try:
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "sector_classification_current" not in names:
            return {}
        marks = {}
        for name, table in (
            ("current", "sector_classification_current"),
            ("revision", "sector_classification_revision"),
        ):
            marks[name] = (
                connection.execute(
                    f"SELECT [count(*), bit_xor(hash(t))] FROM {table} t"
                ).fetchone()[0]
                if table in names
                else None
            )
        return marks
    finally:
        connection.close()


def admit_sectors(database: Path, workspace_id: str) -> dict[str, Any] | None:
    """Admit a workspace's legacy sector classifications after its market listings."""
    port = history.store()
    if port is None or not database.is_file():
        return None
    run = port.lake.execute
    run(f"ATTACH '{sql_path(database)}' AS {WORKSPACE} (READ_ONLY)").complete()
    try:
        tables = {
            row[0]
            for row in run(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = ?", [WORKSPACE]
            ).fetchall()
        }
        if "sector_classification_current" not in tables:
            return None
        current_mark = run(
            f"SELECT [count(*), bit_xor(hash(t))] FROM {WORKSPACE}.sector_classification_current t"
        ).fetchone()[0]
        revision_mark = (
            run(
                "SELECT [count(*), bit_xor(hash(t))] FROM "
                f"{WORKSPACE}.sector_classification_revision t"
            ).fetchone()[0]
            if "sector_classification_revision" in tables
            else None
        )
        marks = {"current": current_mark, "revision": revision_mark}
        revision_columns = {
            row[0]
            for row in run(
                "SELECT column_name FROM information_schema.columns WHERE table_catalog = ?"
                " AND table_name = 'sector_classification_revision'",
                [WORKSPACE],
            ).fetchall()
        }
        forward = {"effective_session", "prior_sector_name", "sector_name"} <= revision_columns
        last = port.log.read(
            "SELECT json_extract(attrs, '$.marks') FROM event WHERE type = 'admission'"
            " AND subject_id = ? AND json_extract(attrs, '$.dataset') = ?"
            " ORDER BY seq DESC LIMIT 1",
            (workspace_id, "sector"),
        )
        if last and json.loads(last[0][0]) == marks:
            return None
        started = port.log.clock()
        providers = [
            row[0]
            for row in run(
                f"SELECT DISTINCT provider FROM {WORKSPACE}.sector_classification_current"
                " ORDER BY 1"
            ).fetchall()
        ]
        rows = 0
        dropped: dict[str, int] = {}
        for provider in providers:
            history.define(
                "provider", provider, f"The {provider} market data provider, as 0.2.7 named it."
            )
            history.define(
                "sector_scheme", provider, f"The sector classification supplied by {provider}."
            )
            scheme = definition_id("sector_scheme", provider)
            alias = definition_id("provider", LISTING_ALIAS)
            prior = (
                "(SELECT arg_min(r.prior_sector_name, r.effective_session)"
                f" FROM {WORKSPACE}.sector_classification_revision r"
                " WHERE r.listing_id = c.listing_id AND r.provider = c.provider"
                " AND r.effective_session IS NOT NULL)"
                if forward
                else "NULL"
            )
            changed = (
                " UNION ALL SELECT a.listing_id, r.effective_session AS valid_from,"
                " r.sector_name AS sector_code"
                f" FROM {WORKSPACE}.sector_classification_revision r"
                " JOIN lake.world.provider_symbol a ON a.symbol = r.listing_id"
                " AND a.provider_id = ?"
                " WHERE r.provider = ? AND r.effective_session IS NOT NULL"
                " AND r.sector_name IS NOT NULL"
                " QUALIFY row_number() OVER (PARTITION BY a.listing_id, r.effective_session"
                " ORDER BY r.retrieved_at DESC) = 1"
                if forward
                else ""
            )
            params = [alias, provider, *([alias, provider] if forward else []), scheme]
            batch = run(
                "WITH facts AS (SELECT a.listing_id, min(l.first_session) AS valid_from,"
                f" coalesce({prior}, c.sector_name) AS sector_code"
                f" FROM {WORKSPACE}.sector_classification_current c"
                " JOIN lake.world.provider_symbol a ON a.symbol = c.listing_id"
                " AND a.provider_id = ?"
                " JOIN lake.world.listing l ON l.listing_id = a.listing_id"
                f" WHERE c.provider = ? GROUP BY ALL{changed})"
                " SELECT ? AS scheme_id, listing_id,"
                " strftime(valid_from, '%Y-%m-%d') AS valid_from,"
                " strftime(lead(valid_from) OVER (PARTITION BY listing_id ORDER BY valid_from)"
                " - INTERVAL 1 DAY, '%Y-%m-%d') AS valid_to,"
                " sector_code FROM facts ORDER BY listing_id, valid_from",
                params,
            ).to_arrow_table()
            expected = run(
                f"SELECT count(*) FROM {WORKSPACE}.sector_classification_current"
                " WHERE provider = ?",
                [provider],
            ).fetchone()[0]
            pc = import_module("pyarrow.compute")  # lazy: the optional data extra

            if len(pc.unique(batch["listing_id"])) != expected:
                raise DataError(
                    "a sector classification has no admitted market listing;"
                    " admit its market data first",
                    code="market_data.sector_listing_missing",
                )
            if batch.num_rows:
                with stage(
                    port.root,
                    "sector",
                    definition_id("provider", provider),
                    batch,
                    lag=MARKET_CLOSE,
                ) as staged:
                    done = port.commit(staged)
                    rows += done.rows
                    for reason, count in done.dropped.items():
                        dropped[reason] = dropped.get(reason, 0) + count
        history.record(
            "admission",
            ("workspace", workspace_id),
            {"provider": providers, "dataset": "sector", "marks": marks, "rows": {"sector": rows}}
            | ({"dropped": dropped} if dropped else {}),
            phase="completed",
            started_us=started,
            node="market_data_ops",
        )
        return {"rows": {"sector": rows}}
    finally:
        run(f"DETACH {WORKSPACE}").complete()


def sector_map(
    listings: Sequence[str], *, session: date, cutoff_us: int, cursor: Any | None = None
) -> dict[str, str]:
    """The unambiguous listing classifications valid at a session and known by a cutoff."""
    port = history.store() if cursor is None else None
    if cursor is None and port is None:
        raise DataError(
            "the Host's sector catalog is not open", code="market_data.sector_catalog_unavailable"
        )
    if not listings:
        return {}
    reader = cursor if cursor is not None else port.lake
    rows = reader.execute(
        "WITH choices AS (SELECT DISTINCT a.symbol, s.scheme_id, s.sector_code, s.valid_from"
        " FROM sector_asof(?) s JOIN provider_symbol_asof(?) a ON a.listing_id = s.listing_id"
        " WHERE a.provider_id = ? AND a.symbol IN (SELECT unnest(?))"
        " AND s.valid_from <= ? AND (s.valid_to IS NULL OR s.valid_to >= ?)),"
        " current AS (SELECT * FROM choices QUALIFY row_number() OVER"
        " (PARTITION BY symbol, scheme_id ORDER BY valid_from DESC) = 1)"
        " SELECT symbol, min(sector_code) FROM current GROUP BY symbol"
        " HAVING count(*) = 1 ORDER BY symbol",
        [
            cutoff_us,
            cutoff_us,
            definition_id("provider", LISTING_ALIAS),
            list(listings),
            session,
            session,
        ],
    ).fetchall()
    return dict(rows)


def sector_holds(database: Path, cursor: Any, *, cutoff_us: int) -> bool:
    """Whether world still holds this workspace's current values and forward revisions."""
    connection = open_workspace_database(database, read_only=True)
    try:
        rows = connection.execute(
            "SELECT listing_id, sector_name FROM sector_classification_current ORDER BY listing_id"
        ).fetchall()
        columns = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'sector_classification_revision'"
            ).fetchall()
        }
        revisions = (
            connection.execute(
                "SELECT listing_id, effective_session, prior_sector_name, sector_name"
                " FROM sector_classification_revision WHERE effective_session IS NOT NULL"
                " ORDER BY effective_session, listing_id, retrieved_at"
            ).fetchall()
            if {"effective_session", "prior_sector_name", "sector_name"} <= columns
            else []
        )
    finally:
        connection.close()
    current = {str(listing): str(name) for listing, name in rows}
    if len(current) != len(rows):
        return False
    if sector_map(tuple(current), session=date.max, cutoff_us=cutoff_us, cursor=cursor) != current:
        return False
    expected = tuple(
        SectorReclassification(str(listing), day, str(prior), str(name))
        for listing, day, prior, name in revisions
        if str(listing) in current
    )
    return sector_transitions(tuple(current), cutoff_us=cutoff_us, cursor=cursor) == expected


def sector_transitions(listings: Sequence[str], *, cutoff_us: int, cursor: Any) -> tuple[Any, ...]:
    """The changes in force for the selected listings among versions known at the cutoff."""
    if not listings:
        return ()
    rows = cursor.execute(
        "WITH facts AS (SELECT DISTINCT a.symbol, s.scheme_id, s.valid_from, s.sector_code"
        " FROM sector_asof(?) s JOIN provider_symbol_asof(?) a ON a.listing_id = s.listing_id"
        " WHERE a.provider_id = ? AND a.symbol IN (SELECT unnest(?))),"
        " changes AS (SELECT symbol, valid_from, sector_code,"
        " lag(sector_code) OVER (PARTITION BY symbol, scheme_id ORDER BY valid_from) AS prior"
        " FROM facts) SELECT symbol, valid_from, prior, sector_code FROM changes"
        " WHERE prior IS NOT NULL AND prior <> sector_code ORDER BY valid_from, symbol",
        [cutoff_us, cutoff_us, definition_id("provider", LISTING_ALIAS), list(listings)],
    ).fetchall()
    return tuple(SectorReclassification(*row) for row in rows)


def _revisions(port: Any, workspace_id: str) -> None:
    """Retain each workspace's original journal row once, at its own observation time."""
    run = port.lake.execute
    for table, (type_, columns) in REVISIONS.items():
        history.define(
            "event_type",
            type_,
            f"One original {table} fact, with its original key and fields.",
            {
                "work": False,
                "subject": table,
                "required": list(columns),
                "text": "none",
                "durable": False,
            },
        )
        held = port.log.read(
            "SELECT subject_id FROM event WHERE type = ? AND workspace_id = ?",
            (type_, workspace_id),
        )
        rows = run(
            f"SELECT {', '.join('r.' + name for name in columns)}, epoch_us(r.observed_at)"
            f" FROM {WORKSPACE}.{table} r WHERE r.revision_id NOT IN"
            " (SELECT unnest(?::VARCHAR[]))"
            " ORDER BY r.observed_at, r.revision_id",
            [[row[0] for row in held]],
        ).fetchall()
        for row in rows:
            attrs = {
                name: value.isoformat() if isinstance(value, date) else value
                for name, value in zip(columns, row[:-1], strict=True)
            }
            history.record_new(
                "SELECT 1 FROM event WHERE type = ? AND subject_id = ? AND workspace_id = ?",
                (type_, attrs["revision_id"], workspace_id),
                type_,
                (table, attrs["revision_id"]),
                attrs,
                occurred_us=row[-1],
                workspace=workspace_id,
                node="market_data_ops",
            )


def _register(run: Any) -> set[str]:
    """Register each provider the tables name, the aliases' and the calendar's, and the lag rule;
    returns the tables' providers."""
    w = WORKSPACE
    named = {
        row[0]
        for row in run(
            f"SELECT provider FROM {w}.market_profile UNION SELECT provider FROM"
            f" {w}.provider_symbol_mapping UNION SELECT provider FROM {w}.raw_daily_bar_current"
            f" UNION SELECT provider FROM {w}.corporate_action_current"
        ).fetchall()
    }
    for name in sorted(named):
        history.define("provider", name, f"The {name} market data provider, as 0.2.7 named it.")
    for name, text in PROVIDERS.items():
        history.define("provider", name, text)
    text = "A 0.2.7 market row with no knowledge time of its own was known at its session's close."
    history.define("lag_rule", MARKET_CLOSE[0], text, {"lag_seconds": 0})
    return named


def _admit(port: Any, providers: set[str]) -> dict[str, Any]:
    """Every table's batches in world's order; the rows each world table took, those StorePort
    dropped by reason, and the later writes of held values it recorded."""
    run, w = port.lake.execute, WORKSPACE
    listings = run(
        "SELECT l.listing_id, l.mic, p.currency, p.calendar_id, p.provider,"
        " strftime(least(min(b.first), min(m.effective_from), min(a.effective_date)), '%Y-%m-%d')"
        f" FROM {w}.listing l JOIN {w}.market_profile p USING (market_profile_id)"
        f" LEFT JOIN (SELECT listing_id, min(session_date) AS first FROM {w}.raw_daily_bar_current"
        " GROUP BY ALL) b USING (listing_id)"
        f" LEFT JOIN {w}.provider_symbol_mapping m USING (listing_id)"
        f" LEFT JOIN {w}.corporate_action_current a USING (listing_id)"
        " GROUP BY ALL ORDER BY l.listing_id"
    ).fetchall()
    dated = [row for row in listings if row[5] is not None]  # a listing with no row of any date
    ids = {row[0]: world_listing(row[0]) for row in dated}
    run("CREATE OR REPLACE TEMP TABLE market_ids (old VARCHAR, new UUID)").complete()
    run("CREATE OR REPLACE TEMP TABLE market_providers (name VARCHAR, id VARCHAR)").complete()
    port.lake.executemany("INSERT INTO temp.market_ids VALUES (?, ?)", list(ids.items()))
    port.lake.executemany(
        "INSERT INTO temp.market_providers VALUES (?, ?)",
        [(name, definition_id("provider", name)) for name in sorted(providers)],
    )
    rows: dict[str, int] = {}
    dropped: dict[str, int] = {}
    rewritten: dict[str, int] = {}

    def commit(table: str, provider: str, batch: Any, lag: Any) -> None:
        if not batch.num_rows:
            return
        lineage = definition_id("provider", provider)
        with stage(port.root, table, lineage, batch, lag=lag) as staged:
            done = port.commit(staged)
        if done.rows:
            rows[table] = rows.get(table, 0) + done.rows
        for reason, count in done.dropped.items():
            dropped[reason] = dropped.get(reason, 0) + count
        if done.rewritten:
            rewritten[table] = rewritten.get(table, 0) + done.rewritten

    for batch in _sessions(run, dated).values():
        commit("session", CALENDARS, _arrow(batch), None)
    for provider, batch in _grouped(
        (
            row[4],
            {
                "listing_id": ids[row[0]],
                "market": market_of(row[3]),
                "universe_class": row[1],
                "currency": row[2],
                "first_session": row[5],
            },
        )
        for row in dated
    ).items():
        commit("listing", provider, _arrow(batch), MARKET_CLOSE)
    for provider, batch in _symbols(run, dated, ids).items():
        commit("provider_symbol", provider, _arrow(batch), MARKET_CLOSE)
    for provider, batch in _bars(port.lake):
        commit("raw_bar", provider, batch, MARKET_CLOSE)
    for provider, batch in _grouped(_actions(run, dated)).items():
        commit("corporate_action", provider, _arrow(batch), MARKET_CLOSE)
    noted = {"dropped": dropped, "rewritten": rewritten}
    return {"rows": rows} | {name: counts for name, counts in noted.items() if counts}


def _arrow(rows: list[dict[str, Any]]) -> Any:
    """A batch built row by row as the Arrow table StorePort stages."""
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    return pa.table(as_columns(rows))


def _grouped(pairs: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for provider, row in pairs:
        out.setdefault(provider, []).append(row)
    return out


def _sessions(run: Any, dated: list[tuple[Any, ...]]) -> dict[str, list[dict[str, Any]]]:
    """Each market's sessions over every valid date the tables hold, from every calendar its
    profiles name; calendars disagreeing on a session are refused by name."""
    codes: dict[str, set[str]] = {}
    for *_, calendar, _, _ in dated:
        codes.setdefault(market_of(calendar), set()).update(re.split("[_+]", calendar))
    if not codes:
        return {}
    w = WORKSPACE
    first, last = run(
        f"SELECT min(d), max(d) FROM (SELECT session_date AS d FROM {w}.raw_daily_bar_current"
        f" UNION ALL SELECT effective_date FROM {w}.corporate_action_current UNION ALL SELECT"
        f" effective_from FROM {w}.provider_symbol_mapping UNION ALL SELECT effective_to FROM"
        f" {w}.provider_symbol_mapping)"
    ).fetchone()
    get_calendar = import_module("exchange_calendars").get_calendar  # lazy: untyped calendars

    start, end = first, last + CALENDAR_REACH
    clocks: dict[str, dict[date, tuple[datetime, datetime]]] = {}
    for market, named in codes.items():
        times: dict[date, tuple[datetime, datetime]] = {}
        for code in sorted(named):
            exchange = get_calendar(code, start=start, end=end)
            clock = exchange.schedule[["open", "close"]]  # the sessions it was built for
            for session, opened, closed in clock.itertuples(index=True, name=None):
                day = session.date()
                pair = (opened.to_pydatetime(), closed.to_pydatetime())  # in UTC, as it gives them
                if times.setdefault(day, pair) != pair:
                    raise DataError(
                        f"the calendars {', '.join(sorted(named))} of market {market} disagree"
                        f" on {day}; give its profiles calendars that agree, then open the"
                        " workspace again",
                        code="market_data.market_calendar_ambiguous",
                    )
        clocks[market] = times
    return {
        market: [
            {
                "market": market,
                "session_date": day.isoformat(),
                "open_utc": opened.isoformat(),
                "close_utc": closed.isoformat(),
                "half_day": closed - opened < FULL_DAY,
            }
            for day, (opened, closed) in sorted(clocks[market].items())
        ]
        for market in sorted(clocks)
    }


def _symbols(
    run: Any, dated: list[tuple[Any, ...]], ids: Mapping[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Each provider's symbol rows, an interval with an end as one row, which StorePort writes as
    its two versions, carrying the source's lifecycle state; an alias keeps the valid_from it
    was first admitted with, so it is never doubled."""
    held = run(
        "SELECT provider_id, CAST(listing_id AS VARCHAR), min(valid_from)::VARCHAR"
        " FROM lake.world.provider_symbol GROUP BY ALL"
    ).fetchall()
    aliases = {(provider, listing): valid for provider, listing, valid in held}
    out: dict[str, list[dict[str, Any]]] = {}
    for listing, provider, symbol, start, end, state in run(
        "SELECT listing_id, provider, provider_symbol, strftime(effective_from, '%Y-%m-%d'),"
        f" strftime(effective_to, '%Y-%m-%d'), lifecycle_state"
        f" FROM {WORKSPACE}.provider_symbol_mapping"
        " ORDER BY listing_id, provider, effective_from"
    ).fetchall():
        if listing in ids:
            row = {"listing_id": ids[listing], "symbol": symbol, "valid_from": start}
            out.setdefault(provider, []).append(row | {"valid_to": end, "lifecycle_state": state})
    display = dict(run(f"SELECT listing_id, display_symbol FROM {WORKSPACE}.listing").fetchall())
    for listing, *_, first in dated:
        for alias, symbol in ((LISTING_ALIAS, listing), (DISPLAY_ALIAS, display[listing])):
            valid = aliases.get((definition_id("provider", alias), ids[listing]), first)
            row = {"listing_id": ids[listing], "symbol": symbol, "valid_from": valid}
            out.setdefault(alias, []).append(row | {"valid_to": None})
    return out


def _bars(lake: Any) -> Iterator[tuple[str, Any]]:
    """Each provider's bars whose values differ from world's newest version of their key, or
    whose source wrote them after that version's value was last written, with the provider's
    adjusted close and their source's last write, in whole years of at most ADMISSION_ROWS bars,
    as DuckDB's Arrow result."""
    w = WORKSPACE
    lake.execute(
        "CREATE OR REPLACE TEMP TABLE market_bars AS SELECT r.provider,"
        " CAST(i.new AS VARCHAR) AS listing_id, r.session_date, r.open, r.high, r.low, r.close,"
        " CAST(r.volume AS DOUBLE) AS volume, a.adjusted_close,"
        " r.payload_hash AS source_payload_digest,"
        f" {_WRITTEN.format('greatest(r.observed_at, a.updated_at)')} AS source_observed_at"
        f" FROM {w}.raw_daily_bar_current r LEFT JOIN {w}.provider_adjusted_close_current a"
        " USING (listing_id, provider, session_date)"
        " JOIN temp.market_ids i ON i.old = r.listing_id"
        " JOIN temp.market_providers v ON v.name = r.provider"
        f" LEFT JOIN {newest('raw_bar')} x ON x.provider_id = v.id"
        " AND x.listing_id = i.new AND x.session_date = r.session_date"
        " WHERE x.listing_id IS NULL OR x.retracted"
        " OR ROW(x.open, x.high, x.low, x.close, x.volume, x.adjusted_close) IS DISTINCT FROM"
        " ROW(r.open, r.high, r.low, r.close, CAST(r.volume AS DOUBLE), a.adjusted_close)"
        " OR greatest(r.observed_at, a.updated_at) > x.ordered_at"
    ).complete()
    years = lake.execute(
        "SELECT provider, year(session_date), count(*) FROM temp.market_bars GROUP BY ALL"
        " ORDER BY ALL"
    ).fetchall()
    for provider, first, last in admissions(years):
        rows = lake.sql(
            "SELECT * EXCLUDE (provider, session_date),"
            " strftime(session_date, '%Y-%m-%d') AS session_date FROM temp.market_bars"
            " WHERE provider = ? AND year(session_date) BETWEEN ? AND ? ORDER BY ALL",
            params=[provider, first, last],
        ).to_arrow_table()
        yield provider, rows


def admissions(
    years: Iterable[tuple[str, int, int]], bound: int = ADMISSION_ROWS
) -> list[tuple[str, int, int]]:
    """Each admission of bars, as its provider and first and last year, from each provider's bars
    by year in order: consecutive whole years while their bars stay within `bound`, a year over
    it alone."""
    groups: list[tuple[str, int, int, int]] = []
    for provider, year, count in years:
        if groups and groups[-1][0] == provider and groups[-1][3] + count <= bound:
            groups[-1] = (provider, groups[-1][1], year, groups[-1][3] + count)
        else:
            groups.append((provider, year, year, count))
    return [(provider, first, last) for provider, first, last, _ in groups]


def _actions(run: Any, dated: list[tuple[Any, ...]]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each corporate action whose values differ from world's newest version of its key, or
    whose source wrote it after that version's value was last written, a retracted one as its
    withdrawal."""
    currency = {row[0]: row[2] for row in dated}
    for action in run(
        "SELECT c.listing_id, c.provider, strftime(c.effective_date, '%Y-%m-%d'), c.action_kind,"
        " c.new_shares_per_old_share, c.cash_amount, c.provisional, c.provenance, c.status,"
        f" {_WRITTEN.format('c.observed_at')}, c.payload_hash, CAST(i.new AS VARCHAR)"
        f" FROM {WORKSPACE}.corporate_action_current c"
        " JOIN temp.market_ids i ON i.old = c.listing_id"
        " JOIN temp.market_providers v ON v.name = c.provider"
        f" LEFT JOIN {newest('corporate_action')} x ON x.provider_id = v.id"
        " AND x.listing_id = i.new AND x.effective_date = c.effective_date"
        " AND x.action_id = c.action_kind"
        " WHERE x.listing_id IS NULL OR ROW(x.ratio, x.amount, x.provisional, x.provenance,"
        " x.retracted) IS DISTINCT FROM ROW(c.new_shares_per_old_share, c.cash_amount,"
        " c.provisional, c.provenance, c.status = 'RETRACTED') OR c.observed_at > x.ordered_at"
        " ORDER BY ALL"
    ).fetchall():
        yield (
            action[1],
            world_action(
                action[:10], action[11], currency[action[0]], source_payload_digest=action[10]
            ),
        )


def world_action(
    action: Sequence[Any], listing_id: str, currency: str | None, *, source_payload_digest: str
) -> dict[str, Any]:
    """One 0.2.7 action, as (listing, provider, date, kind, ratio, amount, provisional,
    provenance, status, its last write), as the world row of `listing_id`: its amount in the
    profile's `currency`, and a retracted one withdrawing its key. Refused by name: a kind world
    does not name."""
    _, _, day, kind, ratio, amount, provisional, provenance, status, written = action
    if kind not in KINDS:
        raise DataError(
            f"0.2.7 holds a {kind} action, a kind world does not name; open the workspace with"
            " the build that names it",
            code="market_data.action_kind_unknown",
        )
    return {
        "listing_id": listing_id,
        "effective_date": day,
        "kind": KINDS[kind],
        "action_id": kind,
        "ratio": ratio,
        "amount": amount,
        "currency": None if amount is None else currency,
        "provisional": provisional,
        "provenance": provenance,
        "retracted": status == "RETRACTED",
        "source_observed_at": written,
        "source_payload_digest": source_payload_digest,
    }
