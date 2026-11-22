"""HTTP requests under the installed outbound source policies."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from typing import Protocol, Self
from urllib.parse import urlsplit

from .network import offline


class HttpResponse(Protocol):
    """The response operations consumed by outbound source readers."""

    @property
    def status_code(self) -> int:
        """Return the response's HTTP status."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Return the response headers."""
        ...

    @property
    def content(self) -> bytes:
        """Return the buffered response body."""
        ...

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield the decoded body in chunks."""
        ...

    def read(self) -> bytes:
        """Read and return the complete response body."""
        ...


class _Request(Protocol):
    @property
    def url(self) -> object: ...


class HttpSource(StrEnum):
    """Installed sources that can open an outbound HTTP session."""

    MODEL_HUB = "model-hub"
    SEC = "sec"
    UNIVERSE = "universe"
    YAHOO_FINANCE = "yfinance"


_SOURCE_HOSTS = {
    HttpSource.MODEL_HUB: frozenset(
        {
            "huggingface.co",
            "cdn-lfs.huggingface.co",
            "cdn-lfs.hf.co",
            "cdn-lfs-us-1.hf.co",
            "cas-bridge.xethub.hf.co",
        }
    ),
    HttpSource.SEC: frozenset({"data.sec.gov", "www.sec.gov"}),
    HttpSource.UNIVERSE: frozenset({"en.wikipedia.org"}),
    HttpSource.YAHOO_FINANCE: frozenset(
        {
            "finance.yahoo.com",
            "query1.finance.yahoo.com",
            "query2.finance.yahoo.com",
            "fc.yahoo.com",
            "guce.yahoo.com",
            "consent.yahoo.com",
        }
    ),
}
"""Exact HTTPS hosts. Hub files also resolve through its existing LFS and Xet delivery hosts."""

# Connect, then read/write/pool. SEC callers can lower or raise their existing request timeout.
_SOURCE_TIMEOUTS = {
    HttpSource.MODEL_HUB: (30.0, 60.0),
    HttpSource.SEC: (15.0, 15.0),
    HttpSource.YAHOO_FINANCE: (30.0, 30.0),
    HttpSource.UNIVERSE: (10.0, 30.0),
}


class EgressRefusal(ValueError):
    """A named refusal checked before the transport reaches the network."""

    def __init__(self, code: str, detail: str) -> None:
        """Carry the stable code and its source into the refusal."""
        self.failure_code = code
        super().__init__(f"{code}: {detail}")


class EgressTransportError(RuntimeError):
    """A transport failure while opening or reading the response."""


class EgressTimeout(EgressTransportError, TimeoutError):
    """A transport time bound expired."""


def require_network(source: HttpSource) -> None:
    """Refuse an outbound source by name when networking is disabled."""
    if offline():
        raise EgressRefusal("egress.network_disabled", source.value)


def require_destination(source: HttpSource, url: str) -> None:
    """Admit a source's destination without DNS, including a redirect target."""
    require_network(source)
    try:
        target = urlsplit(url)
        allowed = (
            target.scheme == "https"
            and target.hostname in _SOURCE_HOSTS[source]
            and target.port in (None, 443)
            and target.username is None
            and target.password is None
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise EgressRefusal("egress.host_not_allowed", source.value)


class HttpEgress:
    """Open one timed source session. A transport can be supplied for offline tests."""

    def __init__(
        self,
        source: HttpSource,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
        transport: object | None = None,
    ) -> None:
        """Open a session for the selected source and its timeout."""
        import httpx  # lazy: only a source opening an HTTP session needs its backend.

        connect, read = _SOURCE_TIMEOUTS[source]
        pool = connect if source == HttpSource.UNIVERSE else read
        if timeout_seconds is not None:
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise EgressRefusal("egress.timeout_invalid", source.value)
            connect = pool = read = timeout_seconds
        self._source = source
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(read, connect=connect, pool=pool),
            follow_redirects=source != HttpSource.SEC,
            transport=transport,
            trust_env=False,
            event_hooks={"request": [self._before_send]},
        )

    def _before_send(self, request: _Request) -> None:
        # HTTPX runs the request hook on the normalized URL before every send in its redirect loop.
        require_destination(self._source, str(request.url))

    @contextmanager
    def stream(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> Iterator[HttpResponse]:
        """Read a GET response without buffering its body."""
        import httpx  # lazy: translate backend failures only when reading an outbound response.

        try:
            with self._client.stream("GET", url, headers=headers) as response:
                yield response
        except httpx.TimeoutException as exc:
            raise EgressTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EgressTransportError(str(exc)) from exc

    def get(self, url: str) -> HttpResponse:
        """Read a complete GET response."""
        with self.stream(url) as response:
            response.read()
            return response

    def close(self) -> None:
        """Release the transport and any open connections."""
        self._client.close()

    def __enter__(self) -> Self:
        """Return the open session."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the session after the caller leaves its scope."""
        self.close()
