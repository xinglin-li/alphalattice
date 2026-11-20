"""Raw HTTP round trips to a live Local Web Host, every checked header exposed.

Kept apart from local_web_support's Portfolio fixtures, so a change to those fixtures does not
select every test that only sends a request.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application.web import (
    SESSION_COOKIE,
    SESSION_HEADER,
)


class _ReadRedirects(urllib.request.HTTPRedirectHandler):
    """The launch URL's 303 is read, not followed: its cookie is the evidence."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_ReadRedirects)


def _request(
    session: LocalPortfolioWebSession,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    host: str | None = None,
    origin: str | None = None,
    omit_origin: bool = False,
    content_type: str = "application/json",
    token: str | None = "valid",
    cookie: str | None = "valid",
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    """One raw HTTP round trip, with every header this service checks exposed."""

    web = session.web
    assert web is not None, "the Host serves no web application"
    application = web.application
    base = session.url.rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base}{path}", data=body, method=method)
    request.add_header("Host", host or f"127.0.0.1:{web.bound_port}")
    if origin is not None:
        request.add_header("Origin", origin)
    elif method == "POST" and not omit_origin:
        request.add_header("Origin", f"http://127.0.0.1:{web.bound_port}")
    if body is not None:
        request.add_header("Content-Type", content_type)
    resolved_token = application.session_token if token == "valid" else token
    resolved_cookie = application.session_token if cookie == "valid" else cookie
    if resolved_token is not None:
        request.add_header(SESSION_HEADER, resolved_token)
    if resolved_cookie is not None:
        request.add_header("Cookie", f"{SESSION_COOKIE}={resolved_cookie}")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(session: LocalPortfolioWebSession, path: str, **kwargs: Any) -> dict[str, Any]:
    status, _headers, body = _request(session, path, **kwargs)
    assert status == 200, (status, body[:400])
    return json.loads(body)
