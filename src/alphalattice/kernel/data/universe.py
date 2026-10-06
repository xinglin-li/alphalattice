"""Frozen no-key current-universe sources and deterministic membership manifests."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from importlib import import_module
from typing import Protocol, cast

from alphalattice.kernel.data.enums import LicenseClass, UniverseIndex
from alphalattice.kernel.data.errors import DataProviderError
from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex
from alphalattice.kernel.shared_kernel.environment import offline

Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class UniverseSourceConfig:
    """Frozen index source location and membership table selector."""

    index: UniverseIndex
    uri: str
    header_prefix: tuple[str, ...]
    symbol_column: int
    company_column: int
    selector: str
    license_class: LicenseClass


FROZEN_UNIVERSE_SOURCES = {
    UniverseIndex.SP500: UniverseSourceConfig(
        index=UniverseIndex.SP500,
        uri="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        header_prefix=("Symbol", "Security", "GICS Sector", "GICS Sub-Industry"),
        symbol_column=0,
        company_column=1,
        selector="first table matching frozen header",
        license_class=LicenseClass.REDISTRIBUTABLE,
    ),
    UniverseIndex.NASDAQ100: UniverseSourceConfig(
        index=UniverseIndex.NASDAQ100,
        uri="https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        header_prefix=("Ticker", "Company"),
        symbol_column=0,
        company_column=1,
        selector="first table with Ticker/Company prefix; symbol column 0, company column 1",
        license_class=LicenseClass.REDISTRIBUTABLE,
    ),
    UniverseIndex.DJIA: UniverseSourceConfig(
        index=UniverseIndex.DJIA,
        uri="https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
        header_prefix=("Company", "Exchange", "Symbol"),
        symbol_column=2,
        company_column=0,
        selector="Company/Exchange/Symbol table; symbol column 2; company column 0",
        license_class=LicenseClass.REDISTRIBUTABLE,
    ),
}


@dataclass(frozen=True)
class FetchedUniverseSource:
    """Acquired source bytes with retrieval time and response hash."""

    config: UniverseSourceConfig
    content: bytes
    retrieved_at: datetime
    response_hash: str


class HttpResponse(Protocol):
    """Minimal HTTP response consumed by the universe source fetcher."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...

    def close(self) -> None: ...


Fetcher = Callable[[str], HttpResponse]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table = []
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            if self._table is not None and self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._table:
                self.tables.append(self._table)
                self._table = None
            self._table_depth -= 1


def _guard_network() -> None:
    if offline():
        raise DataProviderError("network access is disabled", code="data.network_disabled")


def _load_httpx_client() -> tuple[_HttpClient, type[BaseException]]:
    httpx_module = import_module("httpx")
    timeout_factory = cast(Callable[..., object], httpx_module.Timeout)
    client_factory = cast(Callable[..., _HttpClient], httpx_module.Client)
    timeout_exception = cast(
        type[BaseException],
        httpx_module.TimeoutException,
    )
    client = client_factory(
        timeout=timeout_factory(connect=10.0, read=30.0, write=30.0, pool=10.0),
        headers={"User-Agent": "AlphaLattice/0.0 current-universe-research"},
        follow_redirects=True,
    )
    return client, timeout_exception


def fetch_universe_source(
    config: UniverseSourceConfig,
    *,
    fetcher: Fetcher | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Sleeper = time.sleep,
) -> FetchedUniverseSource:
    """Fetch and validate a frozen source under bounded retries and network control."""
    _guard_network()
    owned_client: _HttpClient | None = None
    timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,)
    if fetcher is None:
        owned_client, httpx_timeout = _load_httpx_client()
        timeout_errors = (httpx_timeout, TimeoutError)
        active_fetcher = cast(Fetcher, owned_client.get)
    else:
        active_fetcher = fetcher
    try:
        for attempt in range(1, 4):
            try:
                response = active_fetcher(config.uri)
            except timeout_errors as exc:
                if attempt == 3:
                    raise DataProviderError(
                        "universe source timed out",
                        code="data.provider_timeout",
                        retryable=True,
                        cause_chain=(str(exc),),
                    ) from exc
                sleep(float(2 ** (attempt - 1)))
                continue
            if response.status_code == 429:
                retry_after_text = response.headers.get("Retry-After", "0")
                try:
                    retry_after = float(retry_after_text)
                except ValueError:
                    retry_after = 0.0
                if retry_after > 120 or attempt == 3:
                    raise DataProviderError(
                        "universe source rate limited",
                        code="data.rate_limited",
                        retryable=True,
                    )
                sleep(max(retry_after, float(2 ** (attempt - 1))))
                continue
            if response.status_code != 200:
                raise DataProviderError(
                    f"universe source returned HTTP {response.status_code}",
                    code="data.provider_schema_drift",
                )
            content = bytes(response.content)
            parse_membership(config, content)
            return FetchedUniverseSource(
                config=config,
                content=content,
                retrieved_at=clock(),
                response_hash=sha256_hex(content),
            )
        raise AssertionError("bounded source loop did not terminate")
    finally:
        if owned_client is not None:
            owned_client.close()


def parse_membership(
    config: UniverseSourceConfig,
    content: bytes,
) -> tuple[tuple[str, str], ...]:
    """Extract unique symbol and company rows from the selected source table."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataProviderError(
            "universe source is not UTF-8",
            code="data.provider_schema_drift",
        ) from exc
    parser = _TableParser()
    parser.feed(text)
    target: list[list[str]] | None = None
    for table in parser.tables:
        if table and tuple(table[0][: len(config.header_prefix)]) == config.header_prefix:
            target = table
            break
    if target is None:
        raise DataProviderError(
            f"{config.index} membership table schema drifted",
            code="data.provider_schema_drift",
        )
    members: list[tuple[str, str]] = []
    for row in target[1:]:
        required = max(config.symbol_column, config.company_column)
        if len(row) <= required:
            raise DataProviderError(
                f"{config.index} membership row is partial",
                code="data.partial_response",
            )
        symbol = normalize_symbol(row[config.symbol_column])
        company = " ".join(row[config.company_column].split())
        if not symbol or not company:
            raise DataProviderError(
                f"{config.index} membership row is empty",
                code="data.partial_response",
            )
        members.append((symbol, company))
    if not members:
        raise DataProviderError(
            f"{config.index} membership is empty",
            code="data.partial_response",
        )
    if len({symbol for symbol, _company in members}) != len(members):
        raise DataProviderError(
            f"{config.index} membership contains duplicate symbols",
            code="data.symbol_mismatch",
        )
    return tuple(members)


def normalize_symbol(value: str) -> str:
    """Normalize an index ticker to the canonical dotted symbol."""
    return value.strip().upper().replace("/", ".")


def provider_symbol(value: str) -> str:
    """Map a canonical dotted ticker to a provider hyphenated ticker."""
    return normalize_symbol(value).replace(".", "-")
