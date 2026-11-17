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
from datetime import date, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.root import sql_path
from alphalattice.kernel.data_layer.store import as_columns, newest, stage

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
TABLES: Final = (
    "listing",
    "provider_symbol_mapping",
    "market_profile",
    "raw_daily_bar_current",
    "provider_adjusted_close_current",
    "corporate_action_current",
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
_WRITTEN: Final = "strftime({0}, '%Y-%m-%d %H:%M:%S.%f+00')"
"""A 0.2.7 write time, which 0.2.7 keeps in UTC, as StorePort reads a staged time."""


def world_listing(listing_id: str) -> str:
    """A 0.2.7 listing's world id."""
    return str(uuid.uuid5(LISTINGS, listing_id))


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
        marks = {
            table: run(
                f"SELECT [count(*), bit_xor(hash(t))] FROM {WORKSPACE}.{table} t"
            ).fetchone()[0]
            for table in TABLES
        }
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
        history.record(
            "admission",
            ("workspace", workspace_id),
            {"provider": sorted(providers), "dataset": DATASET, "marks": marks} | admitted,
            phase="completed",
            started_us=started,
            node="market_data_ops",
        )
        return admitted
    finally:
        for drop in DROPS:
            run(drop).complete()
        run(f"DETACH {WORKSPACE}").complete()


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
    its two versions; an alias keeps the valid_from it was first admitted with, so it is never
    doubled."""
    held = run(
        "SELECT provider_id, CAST(listing_id AS VARCHAR), min(valid_from)::VARCHAR"
        " FROM lake.world.provider_symbol GROUP BY ALL"
    ).fetchall()
    aliases = {(provider, listing): valid for provider, listing, valid in held}
    out: dict[str, list[dict[str, Any]]] = {}
    for listing, provider, symbol, start, end in run(
        "SELECT listing_id, provider, provider_symbol, strftime(effective_from, '%Y-%m-%d'),"
        f" strftime(effective_to, '%Y-%m-%d') FROM {WORKSPACE}.provider_symbol_mapping"
        " ORDER BY listing_id, provider, effective_from"
    ).fetchall():
        if listing in ids:
            row = {"listing_id": ids[listing], "symbol": symbol, "valid_from": start}
            out.setdefault(provider, []).append(row | {"valid_to": end})
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
    adjusted close and their source's last write, a year at a time, as DuckDB's Arrow result."""
    w = WORKSPACE
    lake.execute(
        "CREATE OR REPLACE TEMP TABLE market_bars AS SELECT r.provider,"
        " CAST(i.new AS VARCHAR) AS listing_id, r.session_date, r.open, r.high, r.low, r.close,"
        " CAST(r.volume AS DOUBLE) AS volume, a.adjusted_close,"
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
        "SELECT DISTINCT provider, year(session_date) FROM temp.market_bars ORDER BY ALL"
    ).fetchall()
    for provider, year in years:
        rows = lake.sql(
            "SELECT * EXCLUDE (provider, session_date),"
            " strftime(session_date, '%Y-%m-%d') AS session_date FROM temp.market_bars"
            " WHERE provider = ? AND year(session_date) = ? ORDER BY ALL",
            params=[provider, year],
        ).to_arrow_table()
        yield provider, rows


def _actions(run: Any, dated: list[tuple[Any, ...]]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each corporate action whose values differ from world's newest version of its key, or
    whose source wrote it after that version's value was last written, a retracted one as its
    withdrawal."""
    currency = {row[0]: row[2] for row in dated}
    for action in run(
        "SELECT c.listing_id, c.provider, strftime(c.effective_date, '%Y-%m-%d'), c.action_kind,"
        " c.new_shares_per_old_share, c.cash_amount, c.provisional, c.provenance, c.status,"
        f" {_WRITTEN.format('c.observed_at')}, CAST(i.new AS VARCHAR)"
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
        yield action[1], world_action(action[:10], action[10], currency[action[0]])


def world_action(action: Sequence[Any], listing_id: str, currency: str | None) -> dict[str, Any]:
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
    }
