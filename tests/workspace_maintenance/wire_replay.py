"""Yahoo and the index lists replayed below the HTTP client, from a scenario file: no network.

The replay answers at libcurl's transfer. ``yfinance._http.new_session`` hands the product a real
curl_cffi session whose per-thread handle is a :class:`ReplayCurl`. curl_cffi still sets every
option, builds the URL, sends and keeps cookies and parses the response. The product's egress
session still checks each destination, timeout and redirect. yfinance still runs its cookie and
crumb flow and parses the chart, and the product's per-acquisition cache still holds the bodies.
Only ``perform`` is the replay's: it writes the scenario's status, headers and body where libcurl
would, or raises the transfer error libcurl would.

A scenario (``wire_scenarios.SCENARIOS``) names a seeded walk per symbol on the product calendar,
the days a test runs, and named cases of events. Each event has the day the provider first knows
it, so a test moves ``WireReplay.today`` and the provider's answers move with it. Index membership
changes go through the universe fetcher seam (``fetch_universe_source(fetcher=...)``).
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd
import yfinance._http
import yfinance.cache
from curl_cffi.curl import Curl, CurlError, CurlInfo, CurlOpt
from curl_cffi.requests import Session
from yfinance.data import YfData

from alphalattice.capabilities.data.calendar import materialize_calendar_schedule
from alphalattice.capabilities.data.universe import FROZEN_UNIVERSE_SOURCES
from alphalattice.control.data_platform.data_decision_rule import (
    SHIPPED_PRECISION,
    SHIPPED_REBASE_FLOOR,
    SHIPPED_REBASE_PRECISION,
)
from alphalattice.control.data_platform.preflight import resolve_trading_session_authority
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import YFinanceMarketDataProvider
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    CurrentUniverseMaintenanceListing,
    MarketDataRepository,
)
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest
from tests.workspace_maintenance.audit_workspace import audit_one
from tests.workspace_maintenance.local_data_provider import reset_yfinance_cache
from tests.workspace_maintenance.wire_scenarios import SCENARIOS

CRUMB = "replay-crumb"
"""What the replayed getcrumb endpoint answers."""
_TIMEOUT = 28
"""libcurl's CURLE_OPERATION_TIMEDOUT, which curl_cffi raises as its Timeout."""
_COOKIE_EXPIRES = int(datetime(2100, 1, 1, tzinfo=UTC).timestamp())
"""The replayed A3 cookie's expiry, fixed far past the machine's clock, which curl's cookie jar
reads: an expiry a year after a scenario's day drops the cookie once that day is a year past."""
Answer = tuple[int, list[tuple[str, str]], bytes, list[bytes]]
"""A transfer's status, headers, body and libcurl cookie changes."""


@dataclass(frozen=True)
class Event:
    """One provider fact or fault, and the day the provider first shows it."""

    kind: str
    symbol: str = ""
    index: str = ""
    ex: date | None = None
    known: date | None = None
    on: date | None = None
    amount: float = 0.0
    ratio: float = 0.0
    scale: float = 0.0
    status: int | str = 0
    retry_after: int | None = None
    count: int = 0
    join: tuple[str, ...] = ()
    leave: tuple[str, ...] = ()
    restated: tuple[date, float] | None = None
    to: str = ""

    @classmethod
    def read(cls, raw: Mapping[str, Any]) -> Event:
        """An event from its scenario row; dates are ISO strings."""
        values = dict(raw)
        for name in ("ex", "known", "on"):
            if name in values:
                values[name] = date.fromisoformat(values[name])
        for name in ("join", "leave"):
            if name in values:
                values[name] = tuple(values[name])
        if "restated" in values:
            values["restated"] = (
                date.fromisoformat(values["restated"]["known"]),
                float(values["restated"]["amount"]),
            )
        return cls(**values)

    def seen(self, today: date) -> bool:
        """Whether the provider shows this fact on `today`."""
        return self.known is None or self.known <= today


@dataclass
class WireReplay:
    """The provider and the index lists as a scenario has them on `today`.

    `requests` records every transfer as (host, path, query); a test reads it to see what the
    client asked. A request the replay does not serve gets a 404 and stays in `requests`.
    """

    symbols: dict[str, float]
    sectors: dict[str, str]
    sessions: tuple[date, ...]
    days: tuple[date, ...]
    events: tuple[Event, ...]
    membership: dict[str, tuple[tuple[str, str], ...]]
    today: date
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    _faults: dict[tuple[str, date], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def load(cls, case: str | None, *, scenario: str = "realistic") -> WireReplay:
        """The scenario file's walk on its exchange calendar with one named case's events
        (`None`: no events), on its first day."""
        raw = SCENARIOS[scenario]
        days = tuple(date.fromisoformat(item) for item in raw["days"])
        schedule = materialize_calendar_schedule(
            (raw["calendar"],),
            start=date.fromisoformat(raw["start"]),
            end=days[-1],
            as_of_timestamp=datetime.combine(days[-1], datetime.max.time(), UTC),
        )
        return cls(
            symbols={symbol: float(price) for symbol, price in raw["symbols"].items()},
            sectors=dict(raw.get("sectors", {})),
            sessions=tuple(row["session_date"] for row in schedule.to_pylist()),
            days=days,
            events=tuple(Event.read(item) for item in raw["cases"][case]) if case else (),
            membership={
                index: tuple((symbol, name) for symbol, name in rows)
                for index, rows in raw["membership"].items()
            },
            today=days[0],
        )

    @property
    def start(self) -> date:
        return self.sessions[0]

    # The transfer -------------------------------------------------------------------------

    def answer(self, method: str, url: str) -> Answer:
        """Status, headers, body and libcurl cookie changes for one transfer, or a transfer
        error libcurl would raise."""
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        host, path = parts.hostname or "", parts.path
        with self._lock:
            self.requests.append((host, path, query))
        if method != "GET":
            return 405, [], b"", []
        if host == "fc.yahoo.com":
            cookie = f".yahoo.com\tTRUE\t/\tTRUE\t{_COOKIE_EXPIRES}\tA3\treplay-cookie".encode()
            return 404, [("Content-Type", "text/html")], b"<html></html>", [b"SET\t" + cookie]
        if host.endswith("finance.yahoo.com") and path == "/v1/test/getcrumb":
            return 200, [("Content-Type", "text/plain")], CRUMB.encode(), []
        if host.endswith("finance.yahoo.com") and path.startswith("/v10/finance/quoteSummary/"):
            return self._profile(path.rsplit("/", 1)[-1])
        if host.endswith("finance.yahoo.com") and path == "/v7/finance/quote":
            body: dict[str, Any] = {"quoteResponse": {"result": [], "error": None}}
            return 200, [("Content-Type", "application/json")], json.dumps(body).encode(), []
        if host.endswith("finance.yahoo.com") and path.startswith(
            "/ws/fundamentals-timeseries/v1/finance/timeseries/"
        ):
            body = {"timeseries": {"result": [], "error": None}}
            return 200, [("Content-Type", "application/json")], json.dumps(body).encode(), []
        if host.endswith("finance.yahoo.com") and path.startswith("/v8/finance/chart/"):
            symbol = path.rsplit("/", 1)[-1]
            fault = self._fault(symbol=symbol, url=url)
            if fault is not None:
                return fault
            if self._history_symbol(symbol) not in self.symbols:
                body = {"chart": {"result": None, "error": {"code": "Not Found"}}}
                return 404, [("Content-Type", "application/json")], json.dumps(body).encode(), []
            body = {"chart": {"result": [self.chart(symbol, query)], "error": None}}
            return 200, [("Content-Type", "application/json")], json.dumps(body).encode(), []
        return 404, [("Content-Type", "text/html")], b"<html>not replayed</html>", []

    def _fault(self, *, symbol: str = "", index: str = "", url: str = "") -> Answer | None:
        """An outage's error or a move's redirect (to `to`'s host, with the whole query, since
        the product drops the request's parameters when it follows) for the day's first
        `count` requests."""
        for event in self.events:
            if event.kind not in ("outage", "redirect") or event.on != self.today:
                continue
            if (symbol and event.symbol != symbol) or (index and event.index != index):
                continue
            moved = urlsplit(url)
            if event.kind == "redirect" and moved.hostname == event.to:
                continue
            key = (symbol or index, self.today)
            with self._lock:
                used = self._faults.get(key, 0)
                if used >= event.count:
                    return None
                self._faults[key] = used + 1
            if event.kind == "redirect":
                return 302, [("Location", moved._replace(netloc=event.to).geturl())], b"", []
            if event.status == "timeout":
                raise CurlError(
                    "Failed to perform, curl: (28) Operation timed out after 30000 milliseconds "
                    "with 0 bytes received.",
                    _TIMEOUT,
                )
            headers = [("Content-Type", "text/html")]
            if event.retry_after is not None:
                headers.append(("Retry-After", str(event.retry_after)))
            reason = b"Too Many Requests" if event.status == 429 else b"Will be right back"
            return int(event.status), headers, b"<html>" + reason + b"</html>", []
        return None

    def _profile(self, symbol: str) -> Answer:
        """Yahoo's quote summary with the symbol's current sector, or its 404 for a symbol the
        scenario names no sector for."""
        sector = self.sectors.get(symbol)
        if sector is None:
            body = {"quoteSummary": {"result": None, "error": {"code": "Not Found"}}}
            return 404, [("Content-Type", "application/json")], json.dumps(body).encode(), []
        profile = {"sector": sector, "sectorKey": sector.lower().replace(" ", "-")}
        body = {"quoteSummary": {"result": [{"assetProfile": profile}], "error": None}}
        return 200, [("Content-Type", "application/json")], json.dumps(body).encode(), []

    # The chart ------------------------------------------------------------------------------

    def _history_symbol(self, symbol: str) -> str:
        for event in self.events:
            if event.kind == "ticker-change" and event.to == symbol and event.on <= self.today:
                return event.symbol
        return symbol

    def chart(self, symbol: str, query: Mapping[str, str]) -> dict[str, Any]:
        """Yahoo's daily chart result for `symbol` as the provider shows it on `today`."""
        history_symbol = self._history_symbol(symbol)
        bars = self.history(history_symbol, self.today)
        if query.get("range") == "1d":
            sessions = list(bars)[-1:]
        else:
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            first = (epoch + timedelta(seconds=int(query["period1"]))).date()
            last = (epoch + timedelta(seconds=int(query["period2"]))).date()
            sessions = [item for item in bars if first <= item < last]
        for event in self.events:
            if event.kind == "ticker-change" and event.symbol == symbol and event.on <= self.today:
                sessions = [session for session in sessions if session < event.on]
            if event.symbol == symbol and event.kind == "cash-acquisition":
                if event.on is not None and self.today >= event.on:
                    sessions = []
                elif event.ex in sessions:
                    for field in ("open", "high", "low", "close", "adjclose"):
                        bars[event.ex][field] = event.amount
        fault = next(
            (
                event
                for event in self.events
                if event.symbol == symbol
                and event.on == self.today
                and event.kind.startswith("chart-")
            ),
            None,
        )
        if fault is not None and fault.kind == "chart-empty":
            sessions = []
        stamps = [_open_stamp(item) for item in sessions]
        quote = {
            name: [bars[item][name] for item in sessions]
            for name in ("open", "high", "low", "close", "volume")
        }
        adjusted = [bars[item]["adjclose"] for item in sessions]
        if fault is not None and self.today in sessions:
            index = sessions.index(self.today)
            if fault.kind == "chart-null-close":
                quote["close"][index] = None
            elif fault.kind == "chart-zero-close":
                quote["close"][index] = 0.0
        if fault is not None and fault.kind == "chart-partial" and self.today not in sessions:
            stamps.append(_open_stamp(self.today))
            for name, values in quote.items():
                values.append(10 if name == "volume" else bars[self.today][name] * 0.99)
            adjusted.append(bars[self.today]["adjclose"] * 0.99)
        events: dict[str, dict[str, Any]] = {}
        for event, served in self._served_actions(history_symbol, self.today):
            if event.ex not in sessions:
                continue
            stamp = _open_stamp(event.ex)
            if event.kind == "dividend":
                events.setdefault("dividends", {})[str(stamp)] = {"amount": served, "date": stamp}
            else:
                ratio = Fraction(str(event.ratio))
                events.setdefault("splits", {})[str(stamp)] = {
                    "date": stamp,
                    "numerator": ratio.numerator,
                    "denominator": ratio.denominator,
                    "splitRatio": f"{ratio.numerator}:{ratio.denominator}",
                }
        last = stamps[-1] if stamps else _open_stamp(self.today)
        return {
            "timestamp": stamps,
            "events": events,
            "meta": {
                "currency": "USD",
                "symbol": symbol,
                "instrumentType": "EQUITY",
                "exchangeName": "NMS",
                "exchangeTimezoneName": "America/New_York",
                "gmtoffset": -14400,
                "timezone": "EDT",
                "priceHint": 2,
                "dataGranularity": "1d",
                "validRanges": ["1d", "max"],
                "regularMarketTime": last,
                "regularMarketPrice": quote["close"][-1] if sessions else None,
            },
            "indicators": {
                "quote": [quote],
                "adjclose": [{"adjclose": adjusted}],
            },
        }

    def _served_actions(self, symbol: str, today: date) -> list[tuple[Event, float]]:
        """The actions the provider shows on `today`, a dividend at its current amount in the
        latest known units and a split at its ratio."""
        known = [
            event
            for event in self.events
            if event.symbol == symbol and event.kind in {"dividend", "split"} and event.seen(today)
        ]
        splits = [event for event in known if event.kind == "split"]
        served = []
        for event in known:
            if event.kind == "split":
                served.append((event, event.ratio))
                continue
            amount = event.amount
            if event.restated is not None and event.restated[0] <= today:
                amount = event.restated[1]
            later = math.prod(split.ratio for split in splits if split.ex > event.ex)
            served.append((event, amount / later))
        return served

    def history(self, symbol: str, today: date) -> dict[date, dict[str, float]]:
        """Each session through `today`: OHLCV in the units after each split it trades past,
        every earlier bar rebased by each split known on `today`, and the adjusted close Yahoo
        derives from them."""
        return _history(self, symbol, today)

    # The index lists -----------------------------------------------------------------------

    def members(self, index: str) -> dict[str, str]:
        """`index`'s members and company names on `today`, after each join and leave so far."""
        members = dict(self.membership[index])
        for event in self.events:
            if event.kind == "membership" and event.index == index and event.on <= self.today:
                members.update({symbol: symbol.title() for symbol in event.join})
                for symbol in event.leave:
                    members.pop(symbol, None)
            if event.kind == "ticker-change" and event.on <= self.today and event.symbol in members:
                members[event.to] = members.pop(event.symbol)
        return members

    def universe_fetch(self, uri: str) -> SimpleNamespace:
        """The universe fetcher seam: an index page as its source shows it on `today`."""
        config = next(item for item in FROZEN_UNIVERSE_SOURCES.values() if item.uri == uri)
        index = config.index.value
        parts = urlsplit(uri)
        with self._lock:
            self.requests.append((parts.hostname or "", parts.path, {}))
        fault = self._fault(index=index)
        if fault is not None:
            status, headers, body, _ = fault
            return SimpleNamespace(status_code=status, headers=dict(headers), content=body)
        members = self.members(index)
        header = "".join(f"<th>{item}</th>" for item in config.header_prefix)
        rows = []
        for symbol, company in sorted(members.items()):
            cells = [""] * max(len(config.header_prefix), config.symbol_column + 1)
            cells[config.symbol_column], cells[config.company_column] = symbol, company
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        page = f"<html><table><tr>{header}</tr>{''.join(rows)}</table></html>".encode()
        return SimpleNamespace(status_code=200, headers={}, content=page)


def _open_stamp(session: date) -> int:
    """A daily bar's Yahoo timestamp: the session's 09:30 in New York."""
    return int(pd.Timestamp(session.isoformat(), tz="America/New_York").timestamp()) + 34_200


def _walk(symbol: str, base: float, count: int) -> list[float]:
    """A deterministic walk of daily moves under half a percent, seeded by the symbol."""
    seed = sum(ord(char) for char in symbol)
    values, value = [], base
    for step in range(count):
        value *= 1.0 + 0.004 * math.sin(1.7 * step + seed)
        values.append(value)
    return values


def _history(replay: WireReplay, symbol: str, today: date) -> dict[date, dict[str, float]]:
    sessions = [item for item in replay.sessions if item <= today]
    value = _walk(symbol, replay.symbols[symbol], len(sessions))
    for event in replay.events:  # a dividend leaves the price on its ex day, known or not
        if event.kind == "dividend" and event.symbol == symbol and event.ex in sessions:
            drop = event.amount
            for index in range(sessions.index(event.ex), len(sessions)):
                value[index] -= drop
    served = replay._served_actions(symbol, today)
    # A split moves the traded price and volume on its ex day, known or not; once the provider
    # shows it, it rebases every bar before it too.
    units = [
        math.prod(
            event.ratio
            for event in replay.events
            if event.kind == "split"
            and event.symbol == symbol
            and (event.ex <= session or event.seen(today))
        )
        for session in sessions
    ]
    close = [item / unit for item, unit in zip(value, units, strict=True)]
    adjusted = list(close)
    for event, amount in served:
        if event.kind != "dividend" or event.ex not in sessions or event.ex == sessions[0]:
            continue
        ex = sessions.index(event.ex)
        factor = 1.0 - amount / close[ex - 1]
        for index in range(ex):
            adjusted[index] *= factor
    for event in replay.events:
        if event.kind == "reround" and event.symbol == symbol and event.seen(today):
            # The history before the day it is first shown, once and for every later day.
            shown = event.known or sessions[0]
            adjusted = [
                item * (1.0 + event.scale) if session < shown else item
                for item, session in zip(adjusted, sessions, strict=True)
            ]
    seed = sum(ord(char) for char in symbol)
    bars = {}
    for index, session in enumerate(sessions):
        opened, above, below = _offsets(seed, index)
        bars[session] = {
            "open": close[index] * (1.0 + opened),
            "high": close[index] * (1.0 + above),
            "low": close[index] * (1.0 - below),
            "close": close[index],
            "volume": math.floor(
                (1_000_000 + (seed * 31 + index * 997) % 100_000) * units[index] + 0.5
            ),
            "adjclose": adjusted[index],
        }
    return bars


def _offsets(seed: int, ordinal: int) -> tuple[float, float, float]:
    """A session's open, high and low offsets from its close, seeded by the symbol and the
    session and asymmetric, so no range or money-flow factor is constant across symbols."""
    rng = np.random.default_rng((seed, ordinal))
    above, below = float(rng.uniform(0.002, 0.020)), float(rng.uniform(0.002, 0.020))
    return float(rng.uniform(-below * 0.8, above * 0.8)), above, below


class ReplayCurl(Curl):
    """A libcurl handle whose transfer the replay answers. Every option curl_cffi sets reaches
    the real handle; `perform` writes the answer where libcurl would and never connects."""

    def __init__(self, replay: WireReplay) -> None:
        super().__init__()
        self._replay = replay
        self._options: dict[CurlOpt, Any] = {}
        self._info: dict[CurlInfo, Any] = {}

    def setopt(self, option: CurlOpt, value: Any) -> int:
        self._options[option] = value
        return super().setopt(option, value)

    def perform(self, clear_headers: bool = True, clear_resolve: bool = True) -> None:
        del clear_headers, clear_resolve
        url = self._options[CurlOpt.URL]
        url = url.decode() if isinstance(url, bytes) else str(url)
        method = self._options.get(CurlOpt.CUSTOMREQUEST, b"GET")
        method = method.decode() if isinstance(method, bytes) else str(method)
        self._info = {CurlInfo.EFFECTIVE_URL: url.encode(), CurlInfo.RESPONSE_CODE: 0}
        status, headers, body, cookies = self._replay.answer(method.upper(), url)
        head = f"HTTP/1.1 {status} {'OK' if status == 200 else 'Replayed'}\r\n"
        head += "".join(f"{name}: {value}\r\n" for name, value in headers) + "\r\n"
        sink = self._options.get(CurlOpt.HEADERDATA)
        if isinstance(sink, BytesIO):
            sink.write(head.encode())
        write = self._options.get(CurlOpt.WRITEFUNCTION)
        target = self._options.get(CurlOpt.WRITEDATA)
        if callable(write):
            write(body)
        elif isinstance(target, BytesIO):
            target.write(body)
        self._info = {
            CurlInfo.EFFECTIVE_URL: url.encode(),
            CurlInfo.RESPONSE_CODE: status,
            CurlInfo.COOKIECHANGES: cookies,
            CurlInfo.SIZE_DOWNLOAD_T: len(body),
            CurlInfo.HEADER_SIZE: len(head),
        }

    def getinfo(self, option: CurlInfo) -> Any:
        if option in self._info:
            return self._info[option]
        if option in {CurlInfo.REDIRECT_HISTORY, CurlInfo.COOKIECHANGES}:
            return []
        if option in {CurlInfo.EFFECTIVE_URL, CurlInfo.PRIMARY_IP, CurlInfo.LOCAL_IP}:
            return b""
        if option == CurlInfo.REDIRECT_URL:
            return b""
        if option == CurlInfo.TOTAL_TIME:
            return 0.0
        if option == CurlInfo.HTTP_VERSION:
            return 2
        return 0

    def reset(self) -> None:
        self._options.clear()
        super().reset()


class ReplaySession(Session):
    """curl_cffi's own session, each thread's libcurl handle a :class:`ReplayCurl`."""

    def __init__(self, replay: WireReplay) -> None:
        super().__init__(impersonate="chrome")
        self._replay = replay
        self._handles = threading.local()

    @property
    def curl(self) -> Curl:
        handle = getattr(self._handles, "curl", None)
        if handle is None:
            handle = self._handles.curl = ReplayCurl(self._replay)
        return handle


_YFINANCE_CACHES = (
    (yfinance.cache.get_tz_cache, yfinance.cache._TzDBManager, yfinance.cache.tz_db_proxy),
    (
        yfinance.cache.get_cookie_cache,
        yfinance.cache._CookieDBManager,
        yfinance.cache.Cookie_db_proxy,
    ),
    (yfinance.cache.get_isin_cache, yfinance.cache._ISINDBManager, yfinance.cache.isin_db_proxy),
)
"""yfinance's three persistent caches: each one's singleton, database manager and model proxy."""


def install(request: Any, replay: WireReplay, root: Path) -> None:
    """Answer every yfinance transfer from `replay`: its sessions come from `new_session`, its
    caches live under `root`'s provider cache, its process-wide crumb and cookie start empty, and
    the network control is on so the egress check runs (the replay never connects).

    At the test's teardown (`request`'s), the replay's cache databases are closed, then pytest's
    `monkeypatch` restores all the process-wide yfinance state this changed: the shared session,
    crumb and cookie, and each cache's location, database, model binding and state."""
    monkeypatch = request.getfixturevalue("monkeypatch")
    data = YfData()
    for name, value in (
        ("_session", data._session),
        ("_crumb", None),
        ("_cookie", None),
        ("_cookie_strategy", "basic"),
    ):
        monkeypatch.setattr(data, name, value)
    for cache, manager, proxy in _YFINANCE_CACHES:
        instance = cache()
        for name in ("db", "initialised", "dummy"):
            monkeypatch.setattr(instance, name, getattr(instance, name))
        for name in ("_cache_dir", "_db"):
            monkeypatch.setattr(manager, name, getattr(manager, name))
        monkeypatch.setattr(proxy, "obj", proxy.obj)
    reset_yfinance_cache(root / "provider-cache")  # closes the databases open before it
    request.addfinalizer(_close_yfinance_caches)
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "0")
    monkeypatch.setattr(yfinance._http, "new_session", lambda: ReplaySession(replay))
    data.cache_get.cache_clear()


def _close_yfinance_caches() -> None:
    for _cache, manager, _proxy in _YFINANCE_CACHES:
        if manager._db is not None:
            manager._db.close()


@dataclass
class ReplayedWorkspace:
    """A one-listing (AAPL) workspace hydrated through the replay as a first use does, then kept
    by the product's daily maintenance, each runner built as the Host's coordinator builds it
    under a live Goal: the shipped data rule's precision and rebase agreement and the resolved
    session calendar."""

    replay: WireReplay
    market: MarketDataRepository
    manifest: UniverseManifest
    provider: YFinanceMarketDataProvider

    @classmethod
    def first_use(cls, replay: WireReplay, root: Path) -> ReplayedWorkspace:
        """Hydrate AAPL from the scenario's start through its first day and audit it."""
        acquisition = acquisition_manifest()
        manifest = build_quality_filtered_research_manifest(
            replace(acquisition, profile=replace(acquisition.profile, provider="yfinance")),
            eligible_listing_ids=(LISTING,),
        )
        market = MarketDataRepository(root / "workspace")
        market.bootstrap(manifest)
        provider = YFinanceMarketDataProvider(root / "provider-cache")
        replay.today = first = replay.days[0]
        initial = provider.fetch_hydration(
            listing_id=LISTING, provider_symbol="AAPL", start=replay.start, end=first
        )
        market.apply_validated_batch(
            manifest,
            sanitize_payload(manifest, "yfinance", {"AAPL": initial.daily_rows}, ("AAPL",)),
            ingestion_id="initial",
            observed_at=evening(first),
        )
        audit_one(
            market,
            manifest,
            listing_id=LISTING,
            provider="yfinance",
            actions=initial.actions,
            adjusted_closes=initial.adjusted_closes,
            history_start=replay.start,
            history_end=first,
            requested_as_of=first,
            observed_at=evening(first),
        )
        return cls(replay, market, manifest, provider)

    def daily(
        self, day: date, *, approved: bool = False, observed_at: datetime | None = None
    ) -> CurrentUniverseMaintenanceListing:
        """The day's maintenance with the provider as it stands on `day`, and the listing's state
        after it; `approved`: the person approved the listing's full-history audit."""
        observed_at = observed_at or evening(day)
        self.replay.today = observed_at.date()
        approval = frozenset({LISTING} if approved else ())
        runner = CurrentUniverseMaintenance(
            store=self.market,
            manifest=self.manifest,
            provider=self.provider,
            as_of_session=day,
            trading_session_authority=resolve_trading_session_authority(
                start=self.replay.start, end=day, as_of_timestamp=observed_at
            ),
            max_workers=1,
            full_audit_listing_ids=approval,
            full_history_escalation_listing_ids=approval,
            authorization_identity_listing_ids=approval,
            adjusted_tolerance=SHIPPED_PRECISION,
            rebase_agreement=(SHIPPED_REBASE_PRECISION, SHIPPED_REBASE_FLOOR),
        )
        runner.run(observed_at=observed_at)
        return self.market.current_universe_maintenance_listings(runner.maintenance_id)[0]


LISTING = "listing-aapl"
"""The one listing a replayed workspace keeps: the acquisition manifest's AAPL."""


def evening(day: date) -> datetime:
    """23:00 UTC on `day`: after its close, when a daily runs."""
    return datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=23)
