"""Give the market SDK's native session the shared egress policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

from .http import EgressRefusal, HttpSource, require_destination

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 20
_YAHOO_TIMEOUT_SECONDS = 30.0


def yahoo_session(factory: Callable[[], Any]) -> Any:
    """Keep the SDK's session and cookie jar, so the transport applies to its own requests."""
    session = factory()
    send = session.request
    session.trust_env = False

    def request(method: str, url: str, **arguments: Any) -> Any:
        timeout = arguments.pop("timeout", _YAHOO_TIMEOUT_SECONDS)
        if not isinstance(timeout, int | float) or not math.isfinite(timeout) or timeout <= 0:
            raise EgressRefusal("egress.timeout_invalid", HttpSource.YAHOO_FINANCE.value)
        follow = arguments.pop("allow_redirects", True)
        arguments.pop("proxy", None)
        for _ in range(_MAX_REDIRECTS + 1):
            require_destination(HttpSource.YAHOO_FINANCE, url)
            # Send an empty proxy to libcurl and override host-specific requests defaults.
            arguments["proxies"] = {"all": "", f"https://{urlsplit(url).hostname}": ""}
            response = send(method, url, timeout=timeout, allow_redirects=False, **arguments)
            location = response.headers.get("location")
            if not follow or not location or response.status_code not in _REDIRECT_CODES:
                return response
            target = urljoin(response.url, location)
            if urlsplit(target).netloc != urlsplit(response.url).netloc:
                for name in ("headers", "auth", "cookies"):
                    arguments.pop(name, None)
            if response.status_code in (301, 302, 303) and method.upper() == "POST":
                method = "GET"
                for name in ("data", "json", "files"):
                    arguments.pop(name, None)
            arguments.pop("params", None)
            response.close()
            url = target
        raise EgressRefusal("egress.redirect_limit", HttpSource.YAHOO_FINANCE.value)

    session.request = request
    return session
