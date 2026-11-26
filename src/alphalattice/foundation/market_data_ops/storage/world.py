"""A workspace's market rows published into world, the one store of committed market truth,
through the Host's StorePort.

A workspace's market-data.duckdb is a stage's write buffer: its listings, provider symbols, raw
daily bars, the provider's adjusted closes and corporate actions, in 0.2.7's tables. Each writer
journals what it wrote in its own transaction (duckdb.MARKET_DELTA). When a stage's market data
update ends, and when the Host opens the workspace with journal rows pending (`publish_pending`),
`publish` reads the rows the journal names, and only those, as Arrow through the stage's own
connection, and publishes them into the world tables in one catalog transaction, in order:

1. the sessions of each listing's market, generated reference data from its profile's pinned
   exchange calendar rules, retaining that origin and version and known before every cutoff;
2. the listings, each named by the uuid5 of its 0.2.7 id, on the market its profile's calendar
   names (`market_of`), with 0.2.7's universe class (its `mic`) as data, and their provider
   symbols, each listing with two aliases: its 0.2.7 id (LISTING_ALIAS) and its display symbol
   (DISPLAY_ALIAS);
3. the raw bars with the provider's adjusted close, then the corporate actions, each with its
   source's own last write (`source_observed_at`);
4. each workspace's original bar/action revision as its registered log fact, retaining its
   original key, fields and observation time for that workspace's audit cutoff read;

all but the sessions as legacy batches: the listings, provider symbols and universe members under
MARKET_OBSERVED, the bars and actions under MARKET_CLOSE. Each batch is judged once
(StorePort.publish): only rows whose values differ from world's newest version of their key
become versions, a changed row of a held key only when its source wrote it later, known at the
time its writer observed it. The journal is emptied through what was published once the catalog
has committed it, so a crash between the two publishes the same rows again, every one `same`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from importlib import import_module
from itertools import groupby
from pathlib import Path
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.environment import package_versions
from alphalattice.capabilities.quant.sector_history import SectorReclassification
from alphalattice.control.workspace_runtime.database import (
    hand_off,
    open_workspace_database,
    release_workspace_database,
)
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.chain import canonical, digest
from alphalattice.foundation.data_layer.log import attach
from alphalattice.foundation.data_layer.project import (
    ADMISSION_SUMMARY_TEXT_VERSION,
    lineage,
    require_lineage,
)
from alphalattice.foundation.data_layer.store import KNOWN, StorePort, as_columns, newest, stage
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
from alphalattice.kernel import storage
from alphalattice.kernel.identity import FileDigestTrust, Stamp

from . import market_snapshots

DATASET: Final = "market"
WORKSPACE_MARKET_FORMAT: Final = "alphalattice.world-market.v1"
"""A market profile created by a writer that already admits its observations to world."""
MARKET_CLOSE: Final = ("market-close", 1)
"""The lag rule of the bars and corporate actions, which carry no knowledge time: known at their
session's close (lag_seconds 0)."""
MARKET_OBSERVED: Final = ("market-observed", 1)
"""The lag rule of the listings, provider symbols and universe members: known from the start of
their valid date (UTC), the date their workspace records each observed as of, so a first use reads
at its own moment the reference rows it wrote. Every writer's alike: world keys are shared across
workspaces."""
LAG_RULES: Final = {
    MARKET_CLOSE: (
        "A 0.2.7 market row with no knowledge time of its own was known at its session's close.",
        {"lag_seconds": 0},
    ),
    MARKET_OBSERVED: (
        "A listing, provider symbol or universe member is known from the start of its valid date,"
        " the date its workspace records it observed as of.",
        {"lag_seconds": 0, "from": "valid"},
    ),
}
"""Each lag rule world registers, by name and version, with its text and its StorePort spec."""
LISTING_ALIAS: Final = "alphalattice-0.2.7-listing"
"""A listing's 0.2.7 id, its alias: an identity, a name for a key world already holds
(`world_listing` is its uuid5), so a read translating ids reads it at LATEST, whatever the
cutoff."""
LATEST: Final = 253_402_300_799_999_999
"""A read's own moment as a cutoff, the end of year 9999: all that is known when it runs, since
no knowledge time is later than the read."""
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
"""The 0.2.7 tables a workspace's market data is written in, and a publish reads."""
WORKSPACE: Final = "market_workspace"
"""The name of the in-memory database on the catalog session holding the rows one publish read
(`_delta`); a sector admission attaches the workspace's database under it, read-only."""
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
        identity = universe_lineage(workspace_id, event.market_profile_id)
        if identity is None:
            raise ValueError("universe manifest is not registered")
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
            (identity, event.event_hash),
            UNIVERSE_EVENTS[event.kind],
            ("universe", identity),
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


def universe_lineage(workspace_id: str, profile_id: str) -> str | None:
    """The universe definition id assigned to a workspace's governed profile."""
    return _lineage("universe", universe_name(workspace_id, profile_id))


def universe_set_digest(listing_ids: Iterable[str]) -> str:
    """Name a frozen member set by the canonical sorted world listing IDs."""
    return digest(canonical(sorted({world_listing(listing) for listing in listing_ids})))


def universe_set_id(listing_ids: Iterable[str]) -> str | None:
    """The registered universe lineage of a frozen member set."""
    return _lineage("universe", universe_set_digest(listing_ids))


def admit_member_set(listing_ids: Iterable[str], valid_from: date, provider: str) -> str:
    """Admit a frozen set through StorePort, under MARKET_OBSERVED."""
    listings = sorted(set(listing_ids))
    content = universe_set_digest(listings)
    identity = history.define(
        "universe", content, "A frozen universe member set.", {"content_digest": content}
    )
    if identity is None:
        raise DataError(
            "the Host's universe log is not open", code="data_layer.history_unavailable"
        )
    _commit_universe_members(identity, ((item, valid_from, None) for item in listings), provider)
    return identity


def _commit_universe_members(
    identity: str,
    intervals: Iterable[tuple[str, date, date | None]],
    provider: str,
    known: Sequence[int] = (),
) -> None:
    """Admit one universe's legacy intervals through its registered StorePort, under
    MARKET_OBSERVED; each interval known at its `known` time, in microseconds, when given: when
    its membership was decided."""
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
    rows = [row | _known(at) for row, at in zip(rows, known, strict=True)] if known else rows
    if rows:
        lineage = port.log.require_lineage("provider", provider)
        port.publish("universe_member", lineage, _arrow(rows), MARKET_OBSERVED)


@contextmanager
def _log_connection() -> Iterator[sqlite3.Connection | None]:
    """Read the Host's current batch, or the same data root's committed log outside a Host."""
    host = history.opened_log()
    held = host[0] if host is not None else history.holding()
    if held is not None:
        with held.reading() as con:
            yield con
    else:
        root = history.data_root()
        if (root / "machine" / "log.sqlite").is_file():
            with closing(attach(root)) as con:
                yield con
        else:
            yield None


def _log_rows(sql: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
    with _log_connection() as con:
        return [] if con is None else [tuple(row) for row in con.execute(sql, params)]


def _lineage(family: str, name: str) -> str | None:
    with _log_connection() as con:
        return None if con is None else lineage(con, family, name)


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
    """Read a content-keyed set's members known at the cutoff, named by their 0.2.7 aliases (read
    at LATEST), and verify its world content."""
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
            _lineage("provider", LISTING_ALIAS),
            LATEST,
            _lineage("world_table", "provider_symbol"),
            LATEST,
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


def _after_commit(publish: Callable[[], None]) -> None:
    """Run `publish`, a record naming rows admitted so far, once they are committed
    (`StorePort.after_commit`)."""
    port = history.store()
    if port is None:
        raise DataError(
            "the Host's universe store is not open", code="data_layer.history_unavailable"
        )
    port.after_commit(publish)


def _admitted_together(admissions: Sequence[Callable[[], None]]) -> None:
    """Make `admissions` in one catalog snapshot of their own (`StorePort.admitting`)."""
    port = history.store()
    if port is None:
        raise DataError(
            "the Host's universe store is not open", code="data_layer.history_unavailable"
        )
    with port.admitting():
        for admission in admissions:
            admission()


def admit_universe_bootstrap(record: Any, provider: str, workspace_id: str) -> None:
    """Record one frozen bootstrap, with its cohort held as a world member set: the record, which
    names the set and its universe's lineage, once the set is committed (`StorePort.after_commit`),
    after the version that mints that lineage. One recorded with other content is refused now."""
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
    name = universe_name(workspace_id, record.market_profile_id)

    def unrecorded(con: sqlite3.Connection) -> str | None:
        """The universe's lineage while the bootstrap is unrecorded, else None."""
        identity = require_lineage(con, "universe", name)
        held = con.execute(
            "SELECT attrs FROM event WHERE type = 'universe.bootstrapped' AND subject_id = ?",
            (identity,),
        ).fetchall()
        if held and (len(held) != 1 or json.loads(held[0][0]) != attrs):
            raise ValueError("universe bootstrap is already recorded with different content")
        return None if held else identity

    with log.unit() as con:  # a universe with no lineage yet holds no record to disagree with
        if lineage(con, "universe", name) is not None and unrecorded(con) is None:
            return

    def publish() -> None:
        with log.unit() as con:
            identity = unrecorded(con)
            if identity is not None:
                log.record(
                    "universe.bootstrapped",
                    ("universe", identity),
                    attrs,
                    occurred_us=(record.admitted_at - epoch) // timedelta(microseconds=1),
                    node="market_data_ops",
                )

    _after_commit(publish)


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
    """Register an immutable manifest version and its content-keyed world member set: the
    version, which names the set, once the set is committed. A revision registered with other
    content is refused now."""
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

    def unregistered(con: sqlite3.Connection) -> int | None:
        """The version the revision would take, or None when it is registered."""
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
                return None
        return len(versions) + 1

    with log.unit() as con:
        if unregistered(con) is None:
            return set_id

    def publish() -> None:
        with log.unit() as con:
            version = unregistered(con)
            if version is None:
                return
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

    _after_commit(publish)
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
    profile_market = market_of(json.loads(profile[0][0])["profile"]["calendar_id"])
    roster = sorted(set(bootstrap.cohort_listing_ids) | set(reference.admitted_listing_ids))
    days = list(reference.sessions)
    rows = connection.execute(
        "WITH days AS (SELECT unnest(?::DATE[]) AS day),"
        " clocks AS (SELECT s.session_date, max(s.close_utc) AS close_utc FROM lake.world.session s"
        " JOIN (SELECT DISTINCT l.market_id FROM lake.world.listing l"
        " JOIN provider_symbol_asof(?) p USING (listing_id)"
        " JOIN lake.world.market m USING (market_id) WHERE p.provider_id = ?"
        " AND p.symbol IN (SELECT unnest(?::VARCHAR[])) AND m.mic = ?"
        " AND NOT l.retracted AND l.known_asof <= make_timestamptz(?) AND NOT EXISTS"
        " (SELECT 1 FROM lake.world.superseded z WHERE z.table_id = ?"
        " AND z.version_id = l.version_id AND z.superseded_asof <= make_timestamptz(?))) m"
        " USING (market_id)"
        " WHERE NOT s.retracted AND s.known_asof <= make_timestamptz(?) AND NOT EXISTS"
        " (SELECT 1 FROM lake.world.superseded z WHERE z.table_id = ?"
        " AND z.version_id = s.version_id AND z.superseded_asof <= make_timestamptz(?))"
        " GROUP BY s.session_date),"
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
            LATEST,
            _lineage("provider", LISTING_ALIAS),
            roster,
            profile_market,
            cutoff_us,
            _lineage("world_table", "listing"),
            cutoff_us,
            cutoff_us,
            _lineage("world_table", "session"),
            cutoff_us,
            _lineage("provider", LISTING_ALIAS),
            LATEST,
            _lineage("world_table", "provider_symbol"),
            LATEST,
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
    """Admit the bootstrap cohort and its journal as legacy membership intervals, each known
    when its membership was decided, never before it took effect (MARKET_OBSERVED): the cohort's
    when the bootstrap was admitted, an entry's and an exit's when their events were."""
    identity = history.definition_lineage(
        "universe", universe_name(workspace_id, bootstrap.market_profile_id)
    )
    if identity is None:
        raise ValueError("universe manifest is not registered")
    admitted = max(_us(bootstrap.admitted_at), _us(bootstrap.history_start))
    active = {
        listing: (bootstrap.history_start, admitted) for listing in bootstrap.cohort_listing_ids
    }
    intervals: list[tuple[str, date, date | None]] = []
    known: list[int] = []
    for event in sorted(events, key=lambda item: (item.effective_session, item.sequence)):
        decided = max(_us(event.observed_at), _us(event.decided_at), _us(event.effective_session))
        if event.kind == "ENTRY":
            if event.listing_id in active:
                raise ValueError("membership entry for a listing already present")
            active[event.listing_id] = (event.effective_session, decided)
        else:
            if event.listing_id not in active:
                raise ValueError("membership exit for a listing not present")
            start, _ = active.pop(event.listing_id)
            if start < event.effective_session:
                intervals.append((event.listing_id, start, event.effective_session))
                known.append(decided)
    for listing, (start, at) in active.items():
        intervals.append((listing, start, None))
        known.append(at)
    _commit_universe_members(identity, intervals, provider, known)


def admit_universe(database: Path, workspace_id: str) -> None:
    """Admit market and universe data together through their shared admission edge."""
    admit(database, workspace_id)


def carry_workspace(source: Path, database: Path) -> int:
    """Move the workspace whose market database is `database`, published on the data root
    `source`, onto this one: its selection's bodies are written back into its own pending tables
    from `source`'s catalog and the selection dropped (`market_snapshots.write_back`), so this
    root's admission publishes them under its own providers and its next publish selects them
    again. Content and source identity stay; version ids are this root's. A workspace with no
    published selection is left as it is, so a repeated carry writes nothing. No instance of the
    file is kept across it, since one would hold `source`'s catalog. Returns the rows written."""
    end = hand_off([database])
    try:
        with closing(open_workspace_database(database, read_only=False)) as connection:
            if not market_snapshots.anchored(connection):
                return 0
            connection.attach_market_catalog(source)
            return market_snapshots.write_back(connection, database)
    finally:
        end()


def carry_selection(source: Path, database: Path) -> int:
    """Carry into this Host's catalog the world rows that the selection of `database`, a
    workspace's market database or a research-input copy of one, names in the catalog of the data
    root `source` it was published on (`StorePort.carry`): each version as it is, never current
    here, so the selection reads on this root as it read there: a research input keeps every
    version id. A repeated carry adds no row. The recovery a moved research input's refusal names
    (market_data_ops.catalog_selection_unavailable); a workspace a Host serves moves by
    `carry_workspace`. Returns the rows carried: none for a database with no selection."""
    port = history.store()
    if port is None:
        raise ValueError("workspace_database.market_catalog_unavailable")
    with closing(open_workspace_database(database, read_only=True)) as connection:
        if not market_snapshots.enabled(connection):
            return 0
        versions = connection.execute(
            "SELECT world_table, CAST(version_id AS VARCHAR) AS version_id"
            f" FROM {market_snapshots.VERSIONS}"
        ).to_arrow_table()
    return int(port.carry(source, versions))


def _admit_universe_contents(
    database: Path, workspace_id: str, connection: Any, scopes: Sequence[str] | None
) -> None:
    """Copy the manifests, bootstrap and journal of the workspace's market profiles `scopes`
    (every profile, None) before its publish is recorded, read through `connection`, the
    publish's own connection of the file."""
    market = UniverseWorkspaceReader(database.parent, through=connection)
    scoped = "" if scopes is None else " WHERE market_profile_id IN (SELECT unnest(?))"
    named = [] if scopes is None else [list(scopes)]
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "universe_manifest" not in tables:
        return
    manifest_ids = [
        row[0]
        for row in connection.execute(
            f"SELECT manifest_id FROM universe_manifest{scoped}"
            " ORDER BY market_profile_id, as_of_date, manifest_id",
            named,
        ).fetchall()
    ]
    profiles = (
        [
            row[0]
            for row in connection.execute(
                f"SELECT market_profile_id FROM universe_bootstrap{scoped}"
                " ORDER BY market_profile_id",
                named,
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
    minted: list[Callable[[], None]] = []
    for profile in profiles:
        bootstrap = market._workspace_universe_bootstrap(profile)
        if bootstrap is None:
            continue
        events = market._workspace_membership_events(profile)
        provider = providers[profile]
        admit_universe_bootstrap(bootstrap, provider, workspace_id)

        def intervals(
            bootstrap: Any = bootstrap, events: Sequence[Any] = events, provider: str = provider
        ) -> None:
            _admit_universe_intervals(bootstrap, events, provider, workspace_id)
            _after_commit(lambda: _universe_journal(events, workspace_id))

        if universe_lineage(workspace_id, profile) is None:
            minted.append(intervals)
        else:
            intervals()
    if minted:
        # the lineage these intervals name is minted by the manifest's version, published at
        # this block's commit: they take a snapshot of their own after it
        _after_commit(lambda: _admitted_together(minted))


def market_of(calendar: str) -> str:
    """A market profile calendar's market: the ISO 10383 MIC its id names first, so XNYS_XNAS (or
    XNAS+XNYS, as older profiles join them) is XNYS (XNAS)."""
    return re.split("[_+]", calendar)[0]


def _record_admission_summary(
    workspace_id: str, providers: Sequence[str], attrs: dict[str, Any], started: int
) -> None:
    """Record the admission with the full provider list bound to its ledger event."""
    body = canonical(providers)
    history.record(
        "admission",
        ("workspace", workspace_id),
        attrs | {"provider": {"count": len(providers), "digest": digest(body)}},
        text=body.decode(),
        version=ADMISSION_SUMMARY_TEXT_VERSION,
        phase="completed",
        started_us=started,
        node="market_data_ops",
    )


def admit(database: Path, workspace_id: str) -> dict[str, Any] | None:
    """Publish every market row the workspace in `database` holds, as a 0.2.7 workspace's
    upgrade does (`publish`, whole), read from the file opened read-only, which it leaves as it
    is, journal included; None with no Host history or no market database."""
    port = history.store()
    if port is None or not database.is_file():
        return None
    release_workspace_database(database)  # an instance of the file this process keeps open
    upgrade = _UPGRADE_ADMISSION.get()
    same_workspace = upgrade is not None and upgrade.database == database
    source = _selection_source(database) if same_workspace else None
    versions: dict[str, list[Any]] = {}

    def remember(table: str, rows: Any) -> None:
        versions.setdefault(table, []).append(rows)

    with closing(open_workspace_database(database, read_only=True)) as connection:
        capture = remember if source is not None and _bodies_unselected(connection) else None
        admitted = publish(
            database, connection, workspace_id, whole=True, emptying=False, capture_versions=capture
        )
    if upgrade is not None and same_workspace:
        upgrade.selection = None
        if versions and source is not None and _selection_source(database) == source:
            admission = _latest_market_admission(port, workspace_id)
            if admission is not None:
                upgrade.selection = _LegacySelection(
                    workspace_id, source, port, _selection_snapshot(port), admission, versions
                )
    return admitted


def _selection_source(database: Path) -> Stamp | None:
    _, main = FileDigestTrust.version(database)
    has_wal, _ = FileDigestTrust.version(Path(f"{database}.wal"))
    return None if has_wal else main


def _selection_snapshot(port: StorePort) -> int:
    return storage.first(port.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()"))[0]


def _latest_market_admission(port: StorePort, workspace_id: str) -> int | None:
    rows = port.log.read(
        "SELECT seq FROM event WHERE type = 'admission' AND subject_id = ?"
        " AND json_extract(attrs, '$.dataset') = ? ORDER BY seq DESC LIMIT 1",
        (workspace_id, DATASET),
    )
    return int(rows[0][0]) if rows else None


@dataclass(frozen=True)
class _LegacySelection:
    """Version IDs from the fresh upgrade's whole admission and its verified file facts."""

    workspace_id: str
    source: Stamp
    port: StorePort
    snapshot: int
    admission: int
    versions: dict[str, list[Any]]

    def install(self, database: Path, workspace_id: str) -> bool:
        """Use the verified admission under the composition's workspace writer lease."""
        if workspace_id != self.workspace_id or history.store() is not self.port:
            return False
        release_workspace_database(database)
        if _selection_source(database) != self.source:
            return False
        # The selected versions and the legacy comparison read the catalog's World schema.
        with self.port.hold_disjoint_catalog(self.snapshot, "world") as snapshot:
            if (
                snapshot is None
                or _latest_market_admission(self.port, workspace_id) != self.admission
            ):
                return False
            with closing(open_workspace_database(database, read_only=False)) as connection:
                if not market_snapshots.enabled(connection):
                    market_snapshots.install(connection)
                _place_selection(
                    self.port,
                    database,
                    connection,
                    self.versions,
                    _journal_head(connection),
                    moving=True,
                    snapshot=snapshot,
                )
        return True


@dataclass
class UpgradeAdmission:
    """One workspace's admission, owned by the thread finishing a fresh first start."""

    database: Path | None
    selection: _LegacySelection | None = None
    verified: bool = False


_UPGRADE_ADMISSION: ContextVar[UpgradeAdmission | None] = ContextVar(
    "market_upgrade_admission", default=None
)


@contextmanager
def capture_upgrade_admission(database: Path | None) -> Iterator[UpgradeAdmission]:
    """Carry one admission to composition in this thread. None disables capture on resume."""
    upgrade = UpgradeAdmission(database)
    token = _UPGRADE_ADMISSION.set(upgrade)
    try:
        yield upgrade
    finally:
        _UPGRADE_ADMISSION.reset(token)


def _journal_head(connection: Any) -> int | None:
    if (
        storage.first(
            connection.execute(
                "SELECT 1 FROM duckdb_tables() WHERE database_name = current_database()"
                " AND table_name = ?",
                [JOURNAL],
            )
        )
        is None
    ):
        return None
    return storage.first(connection.execute(f"SELECT max(seq) FROM {JOURNAL}"))[0]


def publish_pending(database: Path, workspace_id: str) -> dict[str, Any] | None:
    """The Host's open: publish what the writers of the workspace in `database` journaled and
    world has not published (`publish`), a stage's writes committed before a crash or writes made
    outside a Host since its last publish. The file is opened read-only to find them and writable
    only when the journal holds rows, which the publish then empties; a workspace this machine's
    log has not published is published whole from the read-only connection, journal left. One
    holding bodies with no published selection (`_bodies_unselected`) is published whole from a
    writable connection, its version index installed if missing: its selection then references
    every body and the copies leave the file, in the transaction that places them. None with no
    Host history, no market database or nothing to publish."""
    if history.store() is None or not database.is_file():
        return None
    with closing(open_workspace_database(database, read_only=True)) as connection:
        held = storage.first(
            connection.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE database_name = current_database()"
                " AND table_name = ?",
                [JOURNAL],
            )
        )[0]
        moving = _bodies_unselected(connection)
        if not moving and not (
            held and storage.first(connection.execute(f"SELECT 1 FROM {JOURNAL} LIMIT 1"))
        ):
            if published(workspace_id):
                return None
            return publish(database, connection, workspace_id, whole=True, emptying=False)
    upgrade = _UPGRADE_ADMISSION.get()
    if moving and upgrade is not None and upgrade.verified and upgrade.database == database:
        selection, upgrade.selection = upgrade.selection, None
        if selection is not None and selection.install(database, workspace_id):
            return None
    with closing(open_workspace_database(database, read_only=False)) as connection:
        if moving and not market_snapshots.enabled(connection):
            market_snapshots.install(connection)
        return publish(database, connection, workspace_id, whole=moving)


def _bodies_unselected(connection: Any) -> bool:
    """Whether the workspace holds market bodies with no published selection to reference them
    instead: a 0.2.7 workspace upgraded (its file left as it was, `admit`), one written back to
    move it (`carry_workspace`), or one with no version index yet. Its next writable whole
    publish selects them and drops the copies (`publish`)."""
    held = storage.first(
        connection.execute(
            "SELECT coalesce(list(table_name), []) FROM duckdb_tables()"
            " WHERE database_name = current_database() AND schema_name = 'main'"
        )
    )[0]
    return not market_snapshots.anchored(connection) and any(
        storage.first(connection.execute(f"SELECT 1 FROM {table} LIMIT 1")) is not None
        for table in DELTA
        if table in held
    )


def published(workspace_id: str) -> bool:
    """Whether this machine's log records a market publish of the workspace."""
    return bool(
        _log_rows(
            "SELECT 1 FROM event WHERE type = 'admission' AND subject_id = ?"
            " AND json_extract(attrs, '$.dataset') = ? LIMIT 1",
            (workspace_id, DATASET),
        )
    )


JOURNAL: Final = "market_delta"
"""The workspace's journal of unpublished market writes (duckdb.MARKET_DELTA)."""
JOURNAL_KINDS: Final = frozenset({"bars", "actions", "reference", "universe"})
"""What a journal row names: raw bars or adjusted closes (world's bars), corporate actions, a
market profile's listings, symbols and manifests, or its universe bootstrap and membership
journal."""
MARKET_KINDS: Final = frozenset({"bars", "actions", "reference"})
"""The journal kinds whose rows the market batches publish (`_admit`); a profile's universe
(`_admit_universe_contents`) follows its reference and universe rows, the revision facts
(`_revisions`) its bars and actions."""


def publish(
    database: Path,
    connection: Any,
    workspace_id: str,
    *,
    whole: bool = False,
    emptying: bool = True,
    capture_versions: Callable[[str, Any], None] | None = None,
) -> dict[str, Any] | None:
    """Publish into world, as `workspace_id`'s and in one catalog transaction, the market rows
    of the workspace in `database` its writers journaled (JOURNAL), then empty the journal
    through the last row it took. The rows are read through `connection`, an open connection of
    the file, as Arrow, only those the journal names (`_delta`); each batch is judged once
    (StorePort.publish) and dated by its writer's observed time. A crash before the catalog
    commits leaves the journal for the next publish; one after, before the journal is emptied,
    publishes the same rows again, each judged `same`, which takes no snapshot. `whole`
    publishes every row the tables hold, as does the first publish of a workspace this machine's
    log has none of (`published`): a 0.2.7 workspace upgraded, or one copied from another
    machine. `emptying` False leaves the journal, for a file the publish only reads (`admit`).
    The rows each world table took,
    those StorePort dropped and the later writes of held values it recorded; None when nothing
    was published: no Host history, no market tables, or no journal row. Refused by name: a
    market whose calendars disagree, and as StorePort refuses."""
    port = history.store()
    if port is None:
        return None
    held = set(
        storage.first(
            connection.execute(
                "SELECT list(table_name) FROM duckdb_tables() WHERE database_name ="
                " current_database()"
            )
        )[0]
        or ()
    )
    if not held >= set(TABLES):
        return None
    through = (
        storage.first(connection.execute(f"SELECT max(seq) FROM {JOURNAL}"))[0]
        if JOURNAL in held
        else None
    )
    whole = whole or not published(workspace_id)
    if through is None and not whole:
        return None
    snapshots = emptying and market_snapshots.VERSIONS in held
    # A writable whole publish selects every row, so every body it references leaves the file.
    moving = whole and snapshots
    versions: dict[str, list[Any]] = {}

    def remember(table: str, rows: Any) -> None:
        if snapshots:
            versions.setdefault(table, []).append(rows)
        if capture_versions is not None:
            capture_versions(table, rows)

    if market_snapshots.VERSIONS in held and market_snapshots.anchored(connection):
        market_snapshots.bind_pending_reads(connection, database, port.root, live=True)
    started = port.log.clock()
    run = port.lake.execute
    try:
        kinds, scopes, observed, days = _delta(port.lake, connection, None if whole else through)
        providers = _register(run)
        with port.admitting():  # one catalog snapshot holds every batch this publish makes
            admitted = (
                _admit(
                    port,
                    providers,
                    observed,
                    days,
                    capture_versions=(
                        remember if snapshots or capture_versions is not None else None
                    ),
                )
                if kinds & MARKET_KINDS
                else {"rows": {}}
            )
            if kinds & {"reference", "universe"}:
                _admit_universe_contents(database, workspace_id, connection, scopes)
            if kinds & {"bars", "actions"}:
                _revisions(port, workspace_id)
    finally:
        for drop in DROPS:
            run(drop).complete()
        run(f"DETACH DATABASE IF EXISTS {WORKSPACE}").complete()
    _record_admission_summary(
        workspace_id,
        sorted(providers),
        {"dataset": DATASET, "journal": through} | admitted,
        started,
    )
    history.flush()
    if (through is not None or moving) and emptying and versions:
        _place_selection(
            port,
            database,
            connection,
            versions,
            through,
            moving=moving,
            snapshot=_selection_snapshot(port),
        )
    elif through is not None and emptying:
        connection.execute(f"DELETE FROM {JOURNAL} WHERE seq <= ?", [through])
    return admitted


def _place_selection(
    port: StorePort,
    database: Path,
    connection: Any,
    versions: Mapping[str, Sequence[Any]],
    through: int | None,
    *,
    moving: bool,
    snapshot: int,
) -> None:
    """Commit the workspace selection, remove only covered bodies and empty its journal."""
    names = dict(
        port.log.read(
            "SELECT definition_id, name FROM definition_lineage WHERE family = 'provider'"
        )
    )
    connection.execute("BEGIN TRANSACTION")
    try:
        market_snapshots.place_versions(
            connection, database, versions, names, world_listing, snapshot, port.log.clock()
        )
        for table, (kind, date_column) in DELTA.items():
            physical = market_snapshots.pending_table(database, table)
            version = "corporate_action" if kind == "actions" else "raw_bar"
            action = " AND p.action_kind = v.action_key" if kind == "actions" else ""
            journaled = (
                ""
                if moving
                else f" AND EXISTS (SELECT 1 FROM {JOURNAL} d WHERE d.seq <= ? AND d.kind = ?"
                " AND p.listing_id = d.listing_id AND p.provider = d.provider"
                f" AND p.{date_column} = d.day)"
            )
            connection.execute(
                f"DELETE FROM {physical} p USING {market_snapshots.VERSIONS} v"
                " WHERE v.world_table = ? AND p.listing_id = v.listing_id"
                f" AND p.provider = v.provider AND p.{date_column} = v.day{action}{journaled}",
                [version] if moving else [version, through, kind],
            )
        if through is not None:
            connection.execute(f"DELETE FROM {JOURNAL} WHERE seq <= ?", [through])
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


DELTA: Final = {
    "raw_daily_bar_current": ("bars", "session_date"),
    "provider_adjusted_close_current": ("bars", "session_date"),
    "corporate_action_current": ("actions", "effective_date"),
}
"""The tables a journal row names rows of by their exact key, listing, provider and day: with
its kind and their date column."""
REVISED: Final = {"bar_revision": "bars", "corporate_action_revision": "actions"}
"""The revision tables a journal row names rows of by their `revision_id`: with its kind."""
OBSERVED_US: Final = "_observed_us"
"""The column a published provider symbol mapping carries: when its journal rows observed it."""


def _delta(
    lake: Any, connection: Any, through: int | None
) -> tuple[set[str], list[str] | None, dict[str, int], dict[date, int] | None]:
    """Fill WORKSPACE, an in-memory database on the catalog session, with the workspace's market
    tables holding only the rows the journal names through `through`, read through `connection`
    as Arrow: the bars, adjusted closes and actions of its exact keys, the revisions of its ids,
    the listings it names or whose profile it names whole, the provider symbol mappings of its
    exact keys or of a profile it names whole, each with its journal rows' latest observed time
    (OBSERVED_US), and every market profile; every row when `through` is None. Its kinds, the
    profiles it scopes (None: every one), each named listing's latest observed time in
    microseconds, its own row's where the journal names it, else its profile's, and each bar or
    action day's earliest (None: every day, as a whole publish). A mapping's time is its exact
    key's where the journal names it, else its profile's."""
    journal = "" if through is None else f"(SELECT * FROM {JOURNAL} WHERE seq <= {int(through)})"
    queries: dict[str, str] = {name: f"SELECT * FROM {name}" for name in (*TABLES[:6], *REVISIONS)}
    queries["provider_symbol_mapping"] = (
        f"SELECT *, NULL::BIGINT AS {OBSERVED_US} FROM provider_symbol_mapping"
    )
    kinds, scopes, observed, days = set(JOURNAL_KINDS), None, {}, None
    if journal:
        named_kinds, scopes = storage.first(
            connection.execute(
                "SELECT list(DISTINCT kind),"
                f" coalesce(list(DISTINCT scope ORDER BY scope) FILTER (scope IS NOT NULL), [])"
                f" FROM {journal}"
            )
        )
        kinds = set(named_kinds)
        whole = (
            "SELECT l.listing_id, NULL::VARCHAR AS provider, NULL::DATE AS day, d.kind,"
            f" d.observed_at, true AS whole FROM listing l JOIN {journal} d"
            " ON d.scope = l.market_profile_id AND d.listing_id IS NULL"
        )
        keyed = (
            f"SELECT listing_id, provider, day, kind, observed_at, false AS whole FROM {journal}"
            " WHERE listing_id IS NOT NULL"
        )
        for table, (kind, column) in DELTA.items():
            queries[table] += (
                f" t WHERE EXISTS (SELECT 1 FROM {journal} d WHERE d.kind = '{kind}'"
                " AND d.listing_id = t.listing_id AND d.provider = t.provider"
                f" AND d.day = t.{column})"
            )
        for table, kind in REVISED.items():
            queries[table] += (
                f" WHERE revision_id IN (SELECT revision_id FROM {journal}"
                f" WHERE kind = '{kind}' AND revision_id IS NOT NULL)"
            )
        queries["listing"] += (
            f" WHERE listing_id IN (SELECT listing_id FROM ({keyed} UNION ALL {whole}))"
        )
        queries["provider_symbol_mapping"] = (  # its exact key's time, or its profile's whole
            "SELECT t.*, epoch_us(coalesce(max(d.observed_at) FILTER (NOT d.whole),"
            f" max(d.observed_at))) AS {OBSERVED_US} FROM"
            f" provider_symbol_mapping t JOIN (SELECT * FROM ({keyed}) WHERE provider IS NOT NULL"
            f" UNION ALL {whole}) d ON d.kind = 'reference' AND d.listing_id = t.listing_id"
            " AND (d.provider IS NULL OR d.provider = t.provider AND d.day = t.effective_from)"
            " GROUP BY ALL"
        )
        listed, times = storage.first(
            connection.execute(
                "SELECT coalesce(list(listing_id), []), coalesce(list(us), []) FROM (SELECT"
                " listing_id,"
                " epoch_us(coalesce(max(observed_at) FILTER (kind = 'reference' AND provider"
                " IS NULL"
                " AND NOT whole), max(observed_at) FILTER (kind = 'reference'), max(observed_at)))"
                f" AS us FROM ({keyed} UNION ALL {whole}) GROUP BY ALL)"
            )
        )
        observed = dict(zip(listed, times, strict=True))
        dated, firsts = storage.first(
            connection.execute(
                "SELECT coalesce(list(day), []), coalesce(list(us), []) FROM (SELECT day,"
                f" epoch_us(min(observed_at)) AS us FROM {journal} WHERE day IS NOT NULL"
                " AND kind IN ('bars', 'actions') GROUP BY ALL)"
            )
        )
        days = dict(zip(dated, firsts, strict=True))
    lake.execute(f"ATTACH ':memory:' AS {WORKSPACE}").complete()
    for table, query in queries.items():
        lake.register("delta_rows", connection.sql(query).to_arrow_table())
        try:
            lake.execute(f"CREATE TABLE {WORKSPACE}.{table} AS FROM delta_rows").complete()
        finally:
            lake.unregister("delta_rows")
    return kinds, scopes, observed, days


def admit_sectors(database: Path, workspace_id: str) -> dict[str, Any] | None:
    """Admit a workspace's legacy sector classifications after its market listings, its two
    sector tables read as Arrow through a connection of the file, which inside a step's retained
    instance is a cursor of it (open_workspace_database), never an attach of the file."""
    port = history.store()
    if port is None or not database.is_file():
        return None
    run = port.lake.execute
    run(f"ATTACH ':memory:' AS {WORKSPACE}").complete()
    try:
        with closing(open_workspace_database(database, read_only=True)) as connection:
            for table in ("sector_classification_current", "sector_classification_revision"):
                if storage.first(
                    connection.execute(
                        "SELECT count(*) FROM duckdb_tables() WHERE database_name ="
                        " current_database()"
                        " AND table_name = ?",
                        [table],
                    )
                )[0]:
                    rows = connection.sql(f"FROM {table}").to_arrow_table()
                    port.lake.register("sector_rows", rows)
                    try:
                        run(f"CREATE TABLE {WORKSPACE}.{table} AS FROM sector_rows").complete()
                    finally:
                        port.lake.unregister("sector_rows")
        tables = {
            row[0]
            for row in run(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = ?", [WORKSPACE]
            ).fetchall()
        }
        if "sector_classification_current" not in tables:
            return None
        current_mark = storage.first(
            run(
                "SELECT [count(*), bit_xor(hash(t))] FROM"
                f" {WORKSPACE}.sector_classification_current t"
            )
        )[0]
        revision_mark = (
            storage.first(
                run(
                    "SELECT [count(*), bit_xor(hash(t))] FROM "
                    f"{WORKSPACE}.sector_classification_revision t"
                )
            )[0]
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
            scheme = port.log.require_lineage("sector_scheme", provider)
            alias = port.log.lineage("provider", LISTING_ALIAS)
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
            expected = storage.first(
                run(
                    f"SELECT count(*) FROM {WORKSPACE}.sector_classification_current"
                    " WHERE provider = ?",
                    [provider],
                )
            )[0]
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
                    port.log.require_lineage("provider", provider),
                    batch,
                    lag=MARKET_CLOSE,
                ) as staged:
                    done = port.commit(staged)
                    rows += done.rows
                    for reason, count in done.dropped.items():
                        dropped[reason] = dropped.get(reason, 0) + count
        _record_admission_summary(
            workspace_id,
            providers,
            {"dataset": "sector", "marks": marks, "rows": {"sector": rows}}
            | ({"dropped": dropped} if dropped else {}),
            started,
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
    (found,) = storage.first(
        reader.execute(
            "WITH choices AS (SELECT DISTINCT a.symbol, s.scheme_id, s.sector_code, s.valid_from"
            " FROM sector_asof(?) s JOIN provider_symbol_asof(?) a ON a.listing_id = s.listing_id"
            " WHERE a.provider_id = ? AND a.symbol IN (SELECT unnest(?))"
            " AND s.valid_from <= ? AND (s.valid_to IS NULL OR s.valid_to >= ?)),"
            " current AS (SELECT * FROM choices QUALIFY row_number() OVER"
            " (PARTITION BY symbol, scheme_id ORDER BY valid_from DESC) = 1)"
            " SELECT map(list(symbol ORDER BY symbol), list(code ORDER BY symbol)) FROM"
            " (SELECT symbol, min(sector_code) AS code FROM current GROUP BY symbol"
            " HAVING count(*) = 1)",
            [
                cutoff_us,
                LATEST,
                _lineage("provider", LISTING_ALIAS),
                list(listings),
                session,
                session,
            ],
        )
    )
    return found or {}


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
        [cutoff_us, LATEST, _lineage("provider", LISTING_ALIAS), list(listings)],
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
        ids = storage.first(
            run(f"SELECT coalesce(list(revision_id), []) FROM {WORKSPACE}.{table}")
        )[0]
        held = port.log.read(
            "SELECT subject_id FROM event WHERE type = ? AND workspace_id = ?"
            " AND subject_id IN (SELECT value FROM json_each(?))",
            (type_, workspace_id, json.dumps(ids)),
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
    for (rule, _version), (text, spec) in LAG_RULES.items():
        history.define("lag_rule", rule, text, spec)
    return named


def _admit(
    port: Any,
    providers: set[str],
    observed: Mapping[str, int],
    days: Mapping[date, int] | None,
    *,
    capture_versions: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    """Every table's batches in world's order, from the rows `_delta` read; the rows each world
    table took, those StorePort dropped by reason, and the later writes of held values it
    recorded. A listing's first session is the earliest of world's and the rows read, since no
    writer deletes a market row. A listing's row and its symbols are known at its own `observed`
    time, by 0.2.7 listing id, and a session at the earliest observation of the rows
    of its day (`days`; a day beyond them, of the calendar's reach, at the latest), where the
    journal names them."""
    run, w = port.lake.execute, WORKSPACE
    named = storage.first(run(f"SELECT coalesce(list(listing_id), []) FROM {w}.listing"))[0]
    run("CREATE OR REPLACE TEMP TABLE market_ids (old VARCHAR, new UUID)").complete()
    port.lake.executemany(
        "INSERT INTO temp.market_ids VALUES (?, ?)", [(old, world_listing(old)) for old in named]
    )
    listings = run(
        "SELECT l.listing_id, l.mic, p.currency, p.calendar_id, p.provider,"
        " strftime(least(min(b.first), min(m.effective_from), min(a.effective_date),"
        " min(h.first)), '%Y-%m-%d')"
        f" FROM {w}.listing l JOIN {w}.market_profile p USING (market_profile_id)"
        f" LEFT JOIN (SELECT listing_id, min(session_date) AS first FROM {w}.raw_daily_bar_current"
        " GROUP BY ALL) b USING (listing_id)"
        f" LEFT JOIN {w}.provider_symbol_mapping m USING (listing_id)"
        f" LEFT JOIN {w}.corporate_action_current a USING (listing_id)"
        " JOIN temp.market_ids i ON i.old = l.listing_id"
        " LEFT JOIN (SELECT listing_id, min(first_session) AS first"
        f" FROM {newest('listing')} WHERE NOT retracted GROUP BY ALL) h ON h.listing_id = i.new"
        " GROUP BY ALL ORDER BY l.listing_id"
    ).fetchall()
    dated = [row for row in listings if row[5] is not None]  # a listing with no row of any date
    ids = {row[0]: world_listing(row[0]) for row in dated}
    latest = max(observed.values(), default=None)
    provider_ids = {
        name: port.log.require_lineage("provider", name)
        for name in sorted(providers | PROVIDERS.keys())
    }
    run("CREATE OR REPLACE TEMP TABLE market_providers (name VARCHAR, id VARCHAR)").complete()
    port.lake.executemany(
        "INSERT INTO temp.market_providers VALUES (?, ?)",
        [(name, provider_ids[name]) for name in sorted(providers)],
    )
    rows: dict[str, int] = {}
    dropped: dict[str, int] = {}
    rewritten: dict[str, int] = {}

    def commit(table: str, provider: str, batch: Any, lag: Any) -> None:
        if not batch.num_rows:
            return
        selected = table in {"raw_bar", "corporate_action"} or (
            table == "provider_symbol" and provider not in {LISTING_ALIAS, DISPLAY_ALIAS}
        )
        receive = partial(capture_versions, table) if capture_versions and selected else None
        if receive is None:
            done = port.publish(table, provider_ids[provider], batch, lag)
        else:
            done = port.publish(table, provider_ids[provider], batch, lag, capture_versions=receive)
        if done.rows:
            rows[table] = rows.get(table, 0) + done.rows
        for reason, count in done.dropped.items():
            dropped[reason] = dropped.get(reason, 0) + count
        if done.rewritten:
            rewritten[table] = rewritten.get(table, 0) + done.rewritten

    sessions = _sessions(run, dated, days)
    markets = {mic: port.market_id(mic, provider_ids[CALENDARS]) for mic in sessions}
    arrow = import_module("pyarrow")  # lazy: the optional data extra, only for admission
    reach = max(days.values(), default=None) if days else None
    for mic, batch in sessions.items():
        table = _arrow(
            [
                row | _known(days.get(date.fromisoformat(row["session_date"]), reach))
                if days is not None
                else row | _known(latest)
                for row in batch
            ]
        )
        table = table.drop(["market"]).append_column(
            "market_id", arrow.repeat(markets[mic], table.num_rows)
        )
        commit("session", CALENDARS, table, None)
    for provider, batch in _grouped(
        (
            row[4],
            {
                "listing_id": ids[row[0]],
                "market_id": markets[market_of(row[3])],
                "universe_class": row[1],
                "currency": row[2],
                "first_session": row[5],
            }
            | _known(observed.get(row[0], latest)),
        )
        for row in dated
    ).items():
        commit("listing", provider, _arrow(batch), MARKET_OBSERVED)
    for provider, batch in _symbols(run, dated, ids, provider_ids, observed).items():
        commit("provider_symbol", provider, _arrow(batch), MARKET_OBSERVED)
    for provider, batch in _bars(port.lake):
        commit("raw_bar", provider, batch, MARKET_CLOSE)
    for provider, batch in _grouped(_actions(run, dated)).items():
        commit("corporate_action", provider, _arrow(batch), MARKET_CLOSE)
    noted = {"dropped": dropped, "rewritten": rewritten}
    return {"rows": rows} | {name: counts for name, counts in noted.items() if counts}


def _us(at: datetime | date) -> int:
    """An instant in integer microseconds since the epoch, a date's its start; a naive instant
    is UTC, as stored."""
    if not isinstance(at, datetime):
        at = datetime(at.year, at.month, at.day)
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return (at - datetime.fromtimestamp(0, UTC)) // timedelta(microseconds=1)


def _known(observed_us: int | None) -> dict[str, int]:
    """A batch row's KNOWN column: its writer's observed time; none without one."""
    return {} if observed_us is None else {KNOWN: observed_us}


def _arrow(rows: list[dict[str, Any]]) -> Any:
    """A batch built row by row as the Arrow table StorePort stages."""
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    return pa.table(as_columns(rows))


def _grouped(pairs: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for provider, row in pairs:
        out.setdefault(provider, []).append(row)
    return out


def _sessions(
    run: Any, dated: list[tuple[Any, ...]], days: Mapping[date, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Each market's sessions over every valid date the tables hold, from every calendar its
    profiles name; with a delta publish's `days`, only those days and the calendar's reach past
    the last, since world holds the sessions between. Calendars disagreeing on a session are
    refused by name."""
    codes: dict[str, set[str]] = {}
    for *_, calendar, _, _ in dated:
        codes.setdefault(market_of(calendar), set()).update(re.split("[_+]", calendar))
    if not codes:
        return {}
    w = WORKSPACE
    first, last = storage.first(
        run(
            f"SELECT min(d), max(d) FROM (SELECT session_date AS d FROM {w}.raw_daily_bar_current"
            f" UNION ALL SELECT effective_date FROM {w}.corporate_action_current UNION ALL SELECT"
            f" effective_from FROM {w}.provider_symbol_mapping UNION ALL SELECT effective_to FROM"
            f" {w}.provider_symbol_mapping)"
        )
    )
    if first is None:  # rows of no date: their markets' sessions are world's already
        return {market: [] for market in sorted(codes)}
    get_calendar = import_module("exchange_calendars").get_calendar  # lazy: untyped calendars
    calendar_version = dict(package_versions(("exchange-calendars",)))["exchange-calendars"]

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
                "calendar_origin": "exchange_calendars",
                "calendar_version": calendar_version,
            }
            for day, (opened, closed) in sorted(clocks[market].items())
            if days is None or day in days or day > last
        ]
        for market in sorted(clocks)
    }


def _symbols(
    run: Any,
    dated: list[tuple[Any, ...]],
    ids: Mapping[str, str],
    providers: Mapping[str, str],
    observed: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Each provider's symbol rows, an interval with an end as one row, which StorePort writes as
    its two versions, carrying the source's lifecycle state; an alias keeps the valid_from it
    was first admitted with, so it is never doubled. A mapping is known at its own journal rows'
    time (OBSERVED_US), an alias at its listing's `observed` time, where the journal names one."""
    latest = max(observed.values(), default=None)
    held = run(
        "SELECT provider_id, CAST(listing_id AS VARCHAR), min(valid_from)::VARCHAR"
        " FROM lake.world.provider_symbol GROUP BY ALL"
    ).fetchall()
    aliases = {(provider, listing): valid for provider, listing, valid in held}
    out: dict[str, list[dict[str, Any]]] = {}
    for listing, provider, symbol, start, end, state, known in run(
        "SELECT listing_id, provider, provider_symbol, strftime(effective_from, '%Y-%m-%d'),"
        f" strftime(effective_to, '%Y-%m-%d'), lifecycle_state, {OBSERVED_US}"
        f" FROM {WORKSPACE}.provider_symbol_mapping"
        " ORDER BY listing_id, provider, effective_from"
    ).fetchall():
        if listing in ids:
            row = {"listing_id": ids[listing], "symbol": symbol, "valid_from": start}
            row |= {"valid_to": end, "lifecycle_state": state}
            at = observed.get(listing, latest) if known is None else known
            out.setdefault(provider, []).append(row | _known(at))
    display = dict(run(f"SELECT listing_id, display_symbol FROM {WORKSPACE}.listing").fetchall())
    for listing, *_, first in dated:
        for alias, symbol in ((LISTING_ALIAS, listing), (DISPLAY_ALIAS, display[listing])):
            valid = aliases.get((providers[alias], ids[listing]), first)
            row = {"listing_id": ids[listing], "symbol": symbol, "valid_from": valid}
            row |= {"valid_to": None}
            out.setdefault(alias, []).append(row | _known(observed.get(listing, latest)))
    return out


def _bars(lake: Any) -> Iterator[tuple[str, Any]]:
    """Each provider's bars the publish read, with the provider's adjusted close, their source's
    last write and, as KNOWN, that write's time, in whole years of at most ADMISSION_ROWS bars, as
    DuckDB's Arrow result; StorePort judges them against world's newest version of their key."""
    w = WORKSPACE
    lake.execute(
        "CREATE OR REPLACE TEMP TABLE market_bars AS SELECT r.provider,"
        " CAST(i.new AS VARCHAR) AS listing_id, r.session_date, r.open, r.high, r.low, r.close,"
        " CAST(r.volume AS DOUBLE) AS volume, a.adjusted_close,"
        " r.payload_hash AS source_payload_digest,"
        f" {_WRITTEN.format('greatest(r.observed_at, a.updated_at)')} AS source_observed_at,"
        f" epoch_us(greatest(r.observed_at, a.updated_at)) AS {KNOWN}"
        f" FROM {w}.raw_daily_bar_current r LEFT JOIN {w}.provider_adjusted_close_current a"
        " USING (listing_id, provider, session_date)"
        " JOIN temp.market_ids i ON i.old = r.listing_id"
        " JOIN temp.market_providers v ON v.name = r.provider"
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
    """Each corporate action the publish read, a retracted one as its withdrawal, known (KNOWN)
    at its source's last write; StorePort judges them against world's newest version of their
    key."""
    currency = {row[0]: row[2] for row in dated}
    for action in run(
        "SELECT c.listing_id, c.provider, strftime(c.effective_date, '%Y-%m-%d'), c.action_kind,"
        " c.new_shares_per_old_share, c.cash_amount, c.provisional, c.provenance, c.status,"
        f" {_WRITTEN.format('c.observed_at')}, c.payload_hash, CAST(i.new AS VARCHAR),"
        " epoch_us(c.observed_at)"
        f" FROM {WORKSPACE}.corporate_action_current c"
        " JOIN temp.market_ids i ON i.old = c.listing_id"
        " JOIN temp.market_providers v ON v.name = c.provider"
        " ORDER BY ALL"
    ).fetchall():
        yield (
            action[1],
            world_action(
                action[:10], action[11], currency[action[0]], source_payload_digest=action[10]
            )
            | _known(action[12]),
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
