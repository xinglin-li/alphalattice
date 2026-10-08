"""One loopback HTTP service in front of the local application, and no more.

The browser is a projection. It holds no filesystem path, no store handle, no
permission, no recovery authority and no number it computed itself: every value
it renders arrives as a typed fact this service read from an owner. That is the
whole security model, and everything below exists to keep it true.

**Loopback only.** The socket binds `127.0.0.1`, so there is no LAN listener to
find. Port `0` is the default: the operating system picks a free one and the
launcher prints the resulting URL, which is safer than guessing a fixed port and
racing whatever already holds it.

**One writer.** The service is the sole application authority in the process. It
holds the workspace session, the dispatcher and the Task registry; a browser tab
is a reader that can ask for commands to be admitted.

**Reads are projections; writes are commands.** `GET` returns typed facts and
never mutates. `POST` carries a typed command envelope and is checked four ways
before the application is touched at all -- `Host`, `Origin`, content type and a
per-launch session token that must arrive in both a cookie and a header. Any one
of those failing refuses before a single owner is called.

**The session is the person's.** The cookie is issued only at the launch URL, which
carries a key the Host holds in memory and prints and opens for the person; the
token is read back only by a request that already holds the cookie. A local
program that knows the port reads the projections, as any reader may, but cannot
take an operation only a person may take (HB, V183).

**Nothing leaves the machine.** Assets are inlined from local files, the CSP
forbids every remote origin including `connect-src`, and there is no font,
analytics, CDN or telemetry request to make. `Cross-Origin-Opener-Policy` and
`X-Frame-Options` keep the page out of frames it did not choose.

The implementation is `http.server` from the standard library. It is the
smallest thing that can serve a handful of routes to one local browser, it adds
no dependency, and a framework here would buy routing sugar in exchange for a
supply chain.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final, Protocol, cast
from urllib.parse import parse_qs, urlsplit

from alphalattice.interface.local_application.cli_contract import (
    CLIENT_HEADER,
    INSTANCE_HEADER,
    MAXIMUM_REQUEST_BODY_BYTES,
    REQUEST_PROVENANCE,
    WORKSPACE_HEADER,
    RequestProvenance,
    refusal_words,
    request_provenance,
)
from alphalattice.interface.local_application.failure_codes import (
    owner_failure_code,
    public_failure,
)
from alphalattice.protocols.research_authoring.selection import rewrite_in_declaration_dialect

LOOPBACK_HOST = "127.0.0.1"
BROWSER_REFUSED_PORTS: Final = frozenset(
    {1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95}
    | {101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161}
    | {179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563}
    | {587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060}
    | {5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080}
)
"""The ports Chromium refuses to load (`ERR_UNSAFE_PORT`). Windows may assign one to a
port-0 bind, as the staged gate met at 1720 (STOPS-1), and the Workbench would not open."""
BROWSER_SAFE_REBINDS: Final = 16
"""How many refused ports one bind holds while it asks for another, before it keeps one."""
SESSION_COOKIE = "alphalattice_local_session"
"""Cookie-name prefix; browsers share cookies across ports on one loopback host."""
SESSION_HEADER = "X-Alphalattice-Session"
LAUNCH_PATH = "/launch"
"""The one route that issues the session cookie, with the Host's launch key."""
REOPEN_FROM_LAUNCH_URL = "REOPEN_FROM_LAUNCH_URL"
"""The way on from a refused session: open the launch URL the Host printed."""


"""The token travels in both a cookie and a header.

A cookie alone is sent by the browser on any cross-site form post, which is the
forgery this is here to stop. A header alone survives, but only same-origin
script can set it -- so requiring both means a page on another origin can supply
at most one of them.
"""


def session_cookie_name(port: int) -> str:
    """Return the cookie name scoped to one loopback service port."""
    return f"{SESSION_COOKIE}_{port}"


# A content-hashed asset never changes under its path (round 96).
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "object-src 'none'",
    )
)
"""`default-src 'none'` and then only what the page actually uses.

`connect-src 'self'` is the load-bearing one: it is what makes "no telemetry" a
property of the page rather than a promise about its author.
"""


class LocalWebError(ValueError):
    """Stable refusal for a malformed or unauthorised local request."""

    def __init__(self, reason: str, *, next_action: str | None = None):
        """Carry a safe refusal reason and optional recovery action."""
        super().__init__(reason)
        self.next_action = next_action


@dataclass(frozen=True, slots=True)
class Route:
    """One method, one path, one handler, and whether it may write."""

    method: str
    path: str
    handler: Callable[[Mapping[str, list[str]], dict[str, Any]], object]
    mutates: bool = False
    external_client: bool = False
    session: bool = False
    """A read that answers only the browser holding the session cookie."""


@dataclass(frozen=True, slots=True)
class LocalWebResponse:
    """Encoded HTTP answer with route-specific security and cache policies."""

    status: HTTPStatus
    body: bytes
    content_type: str
    content_security_policy: str | None = None
    cache_control: str | None = None
    location: str | None = None
    """Where a redirect sends the browser (the launch route only)."""
    """How long a browser may keep this response; `None` is `no-store`.

    Every API answer and the host page are `no-store`: the product's state is
    read from its owners, never from a cache. An asset served under its
    content-hashed path is the one exception (round 96): its bytes cannot
    change under that path, so it is `immutable` for a year and a warm load
    fetches nothing.
    """
    """A route-specific policy, when the default one would break the artifact.

    Used by exactly one route: the sealed report page carries its stylesheet
    inline, and `style-src 'self'` forbids that. The answer is not to relax the
    policy for the whole service -- it is to let that response name the exact
    style it contains, by hash.
    """


@dataclass(frozen=True, slots=True)
class AdmittedRequest:
    """A request that passed every check that can be made without its body.

    Carried rather than recomputed: the handler reads exactly `content_length`
    bytes and hands this back, so the route that was authorised and the route
    that runs cannot be two different routes.
    """

    route: Route | None
    asset: tuple[bytes, str, str | None] | None
    query: Mapping[str, list[str]]
    content_length: int
    issues_session: bool
    launch: bool = False
    provenance: RequestProvenance | None = None


@dataclass
class LocalWebApplication:
    """Routing, authorisation and encoding. It owns no domain logic.

    Handlers receive parsed query values and a parsed JSON payload and return
    plain data; this class turns that into bytes and headers. Anything that
    needed a store, a path or a lock would have to be given one, and nothing here
    can give one.
    """

    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    launch_key: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    """Held in memory and put only in the launch URL; never on disk or in an answer."""
    external_token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    external_instance: str = field(default_factory=lambda: secrets.token_hex(16))
    external_workspace_id: str | None = None
    cookie_name: str = field(default=SESSION_COOKIE, init=False, repr=False)
    routes: dict[tuple[str, str], Route] = field(default_factory=dict)
    assets: dict[str, tuple[bytes, str, str | None]] = field(default_factory=dict)
    allowed_hosts: frozenset[str] = frozenset()
    refusals: int = 0
    served: int = 0

    def route(
        self,
        method: str,
        path: str,
        *,
        mutates: bool = False,
        external_client: bool = False,
        session: bool = False,
    ) -> Callable[
        [Callable[[Mapping[str, list[str]], dict[str, Any]], object]],
        Callable[[Mapping[str, list[str]], dict[str, Any]], object],
    ]:
        """Register a handler under one method, path, and access policy.

        Args:
            method: The HTTP method.
            path: The exact route path.
            mutates: Whether the handler may write product state.
            external_client: Whether the external client may call it.
            session: Whether a read answers only the browser holding the session
                cookie (the session read, which hands that browser its token).

        Returns:
            A decorator that records the handler without changing it.
        """

        def register(
            handler: Callable[[Mapping[str, list[str]], dict[str, Any]], object],
        ) -> Callable[[Mapping[str, list[str]], dict[str, Any]], object]:
            self.routes[(method, path)] = Route(
                method=method,
                path=path,
                handler=handler,
                mutates=mutates,
                external_client=external_client,
                session=session,
            )
            return handler

        return register

    def add_asset(
        self, path: str, body: bytes, content_type: str, *, immutable: bool = False
    ) -> None:
        """Register an asset for local serving.

        A content-hashed path may be immutable; a plain path remains no-store
        so an older link reads the current build.

        Args:
            path: The served asset path.
            body: The asset bytes.
            content_type: The response media type.
            immutable: Whether the path binds the asset's content hash.
        """
        self.assets[path] = (body, content_type, IMMUTABLE_CACHE if immutable else None)

    # ------------------------------------------------------------- security

    def _authorised(
        self,
        *,
        method: str,
        headers: Mapping[str, str],
        mutates: bool,
        external_client: bool = False,
        session: bool = False,
    ) -> str | None:
        """Return a refusal reason, or `None` when the request may proceed.

        Checked in the order a hostile request fails fastest, and *before* any
        handler runs -- an authorisation check after the application has been
        touched is not an authorisation check.
        """
        host = (headers.get("Host") or "").split(",")[0].strip()
        if host not in self.allowed_hosts:
            return "local_web.host_not_allowed"
        origin = headers.get("Origin")
        if origin is not None:
            allowed = {f"http://{value}" for value in self.allowed_hosts}
            if origin not in allowed:
                return "local_web.origin_not_allowed"
        if external_client:
            if origin is not None:
                return "local_web.external_browser_origin_not_admitted"
            if self.external_workspace_id is None:
                return "local_web.external_client_not_admitted"
            if (
                headers.get(WORKSPACE_HEADER) != self.external_workspace_id
                or headers.get(INSTANCE_HEADER) != self.external_instance
            ):
                return "local_web.external_workspace_or_instance_mismatch"
            token = headers.get(CLIENT_HEADER, "")
            if not token or not secrets.compare_digest(token, self.external_token):
                return "local_web.external_token_invalid"
            if (
                mutates
                and headers.get("Content-Type", "").split(";")[0].strip().lower()
                != "application/json"
            ):
                return "local_web.content_type_not_json"
            return None
        if not mutates:
            if session:
                # The session read hands its token to the browser the Host opened,
                # which holds the cookie the launch URL set; no one else has it.
                cookie = _cookie(headers.get("Cookie"), self.cookie_name)
                if cookie is None:
                    # Explicit local clients from before port-scoping may still present
                    # the exact session secret. Browser launches never set this shared name.
                    cookie = _cookie(headers.get("Cookie"), SESSION_COOKIE)
                if cookie is None:
                    return "local_web.session_token_absent"
                if not secrets.compare_digest(cookie.encode(), self.session_token.encode()):
                    return "local_web.session_token_invalid"
            return None
        # Everything below is for writes only.
        if origin is None:
            # A same-origin fetch sends `Origin` for POST. Its absence on a
            # mutation is the signature of a form post from somewhere else.
            return "local_web.origin_absent_on_mutation"
        content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            # Deliberately narrow: the three form encodings are exactly what a
            # cross-site form can send without script.
            return "local_web.content_type_not_json"
        cookie = _cookie(headers.get("Cookie"), self.cookie_name)
        if cookie is None:
            # Keep same-token local clients compatible; never accept a different token.
            cookie = _cookie(headers.get("Cookie"), SESSION_COOKIE)
        supplied = headers.get(SESSION_HEADER)
        if cookie is None or supplied is None:
            return "local_web.session_token_absent"
        if not (
            secrets.compare_digest(cookie, self.session_token)
            and secrets.compare_digest(supplied, self.session_token)
        ):
            return "local_web.session_token_invalid"
        return None

    # -------------------------------------------------------------- serving

    def preflight(
        self, *, method: str, target: str, headers: Mapping[str, str]
    ) -> AdmittedRequest | LocalWebResponse:
        """Everything decidable before a byte of the body is read.

        Route, `Host`, `Origin`, content type, session, CSRF and the declared
        length are all properties of the request line and its headers. Checking
        them here is what makes the bound real: a refusal costs one response,
        and the announced body is never read, never decoded and never allocated.
        """
        parsed = urlsplit(target)
        path = parsed.path
        if method == "GET" and path == LAUNCH_PATH:
            return self._launch(parsed.query, headers)
        asset = self.assets.get(path) if method == "GET" else None
        route = self.routes.get((method, path))
        if asset is None and route is None:
            return self._refuse(HTTPStatus.NOT_FOUND, "local_web.route_unknown")
        refusal = self._authorised(
            method=method,
            headers=headers,
            mutates=bool(route and route.mutates),
            external_client=bool(route and route.external_client),
            session=bool(route and route.session),
        )
        if refusal is not None:
            self.refusals += 1
            return self._refuse(
                HTTPStatus.FORBIDDEN,
                refusal,
                next_action=REOPEN_FROM_LAUNCH_URL if route and route.session else None,
            )
        declared = (headers.get("Content-Length") or "0").strip()
        if not declared.isdigit():
            # `isdigit` rejects the whole malformed family in one test: a sign,
            # a space, a decimal point, a second value from a duplicated header.
            self.refusals += 1
            return self._refuse(HTTPStatus.BAD_REQUEST, "local_web.content_length_invalid")
        content_length = int(declared)
        if content_length > MAXIMUM_REQUEST_BODY_BYTES:
            self.refusals += 1
            return self._refuse(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"local_web.body_too_large:{MAXIMUM_REQUEST_BODY_BYTES}",
            )
        try:
            # An external client may name its agent session and a goal (OP13).
            provenance = (
                request_provenance(headers) if route is not None and route.external_client else None
            )
        except ValueError as error:
            self.refusals += 1
            return self._refuse(HTTPStatus.BAD_REQUEST, str(error))
        return AdmittedRequest(
            route=route,
            asset=asset,
            query=parse_qs(parsed.query),
            content_length=content_length,
            # The session cookie is issued at the launch URL alone (`_launch`).
            issues_session=False,
            provenance=provenance,
        )

    def _launch(self, query: str, headers: Mapping[str, str]) -> AdmittedRequest | LocalWebResponse:
        """The launch URL: with the Host's key, the session cookie and the page.

        The key reaches only the browser the Host opened (and the person who reads
        the printed URL); the answer sets the cookie and sends the browser on to
        the Workbench, which reads its token with that cookie. Any other key, or
        none, is refused with the way on.
        """
        refusal = self._authorised(method="GET", headers=headers, mutates=False)
        keys = parse_qs(query).get("key", [])
        if refusal is None and (
            len(keys) != 1 or not secrets.compare_digest(keys[0].encode(), self.launch_key.encode())
        ):
            refusal = "local_web.launch_key_invalid"
        if refusal is not None:
            self.refusals += 1
            return self._refuse(HTTPStatus.FORBIDDEN, refusal, next_action=REOPEN_FROM_LAUNCH_URL)
        return AdmittedRequest(
            route=None,
            asset=None,
            query={},
            content_length=0,
            issues_session=True,
            launch=True,
        )

    def dispatch(self, admitted: AdmittedRequest, body: bytes) -> LocalWebResponse:
        """Run the route that preflight authorised, over the body it bounded."""
        if admitted.launch:
            return LocalWebResponse(
                HTTPStatus.SEE_OTHER,
                b"",
                "text/plain; charset=utf-8",
                location="/workbench.html",
            )
        if admitted.asset is not None:
            self.served += 1
            content, content_type, cache_control = admitted.asset
            return LocalWebResponse(
                HTTPStatus.OK, content, content_type, cache_control=cache_control
            )
        route = admitted.route
        if route is None:  # pragma: no cover - preflight admits one or the other
            return self._refuse(HTTPStatus.NOT_FOUND, "local_web.route_unknown")
        payload: dict[str, Any] = {}
        if body:
            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._refuse(HTTPStatus.BAD_REQUEST, "local_web.body_not_json")
            if not isinstance(decoded, dict):
                return self._refuse(HTTPStatus.BAD_REQUEST, "local_web.body_not_an_object")
            payload = decoded
        scope = REQUEST_PROVENANCE.set(admitted.provenance)
        try:
            answer = route.handler(admitted.query, payload)
        except LocalWebError as error:
            return self._refuse(
                HTTPStatus.BAD_REQUEST,
                public_failure(error, "local_application.request_refused"),
                next_action=error.next_action,
            )
        except Exception as error:
            # Refusal identity belongs to the owner, not to ValueError's inheritance.
            code = owner_failure_code(error)
            if code is not None:
                return self._refuse(HTTPStatus.BAD_REQUEST, code)
            if isinstance(error, (ValueError, KeyError)):
                return self._refuse(
                    HTTPStatus.BAD_REQUEST,
                    public_failure(error, "local_application.request_refused"),
                )
            # Anything else still has to become a response. An unhandled
            # exception here closes the socket, and a dropped connection tells a
            # local UI nothing at all -- it cannot distinguish a bug from a
            # service that stopped. Its detail stays in the Host's log.
            return self._refuse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                public_failure(error, "local_web.handler_failed"),
            )
        finally:
            REQUEST_PROVENANCE.reset(scope)
        self.served += 1
        if isinstance(answer, LocalWebResponse):
            return answer
        if isinstance(answer, dict) and isinstance(answer.get("yaml"), str):
            # An exported declaration must read back as it was written (V146). Until W12 moves
            # each exporter onto `dump_declaration`, the answer is rewritten here, once.
            answer = {**answer, "yaml": rewrite_in_declaration_dialect(answer["yaml"])}
        return LocalWebResponse(
            HTTPStatus.OK,
            json.dumps(answer, default=str, sort_keys=True).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    @staticmethod
    def _refuse(
        status: HTTPStatus, reason: str, *, next_action: str | None = None
    ) -> LocalWebResponse:
        body: dict[str, object] = {"refused": reason}
        # A raised refusal carries its code's words and way on, as a returned one does (OP4,
        # V449).
        words = refusal_words(reason)
        if next_action is not None or words:
            body.update(
                status="REFUSED",
                failure_code=reason,
                **({"detail": words["detail"]} if words else {}),
                next_action=next_action or words["next_action"],
            )
        return LocalWebResponse(
            status,
            json.dumps(body).encode("utf-8"),
            "application/json; charset=utf-8",
        )


def report_policy(page: str) -> str:
    """The default policy plus the exact inline styles this page contains.

    A sealed report is an artifact: its bytes are the evidence and may not be
    rewritten to carry a nonce. Hashing each `<style>` block gives the browser
    permission for precisely those bytes and nothing else -- narrower than
    `'unsafe-inline'`, which would permit any style anyone later injected.
    """
    digests = [
        "'sha256-" + base64.b64encode(hashlib.sha256(block.encode("utf-8")).digest()).decode() + "'"
        for block in re.findall(r"<style[^>]*>(.*?)</style>", page, flags=re.DOTALL)
    ]
    if not digests:
        return CONTENT_SECURITY_POLICY
    return CONTENT_SECURITY_POLICY.replace(
        "style-src 'self'", "style-src 'self' " + " ".join(digests)
    )


def _cookie(header: str | None, name: str) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "AlphalatticeLocal/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 10.0
    """An idle keep-alive socket must not pin a thread open forever.

    `HTTP/1.1` means a browser keeps its connection, and a handler waiting on a
    connection that will never speak again is a thread shutdown has to join. With
    a socket timeout that wait is bounded, so stopping the service is bounded
    too, whatever the browser is doing.
    """

    disable_nagle_algorithm = True
    """Headers and body are two writes, and Nagle plus delayed ACK can turn that
    into a stall. Kept because nothing here is bandwidth-bound, so there is
    nothing to coalesce and nothing to lose by sending immediately -- but it was
    *not* the cause of the 296 ms round trip it was first added for. That was
    measured after the fact and refuted: the time was in the application, and
    setting this changed nothing on its own."""

    @property
    def application(self) -> LocalWebApplication:
        application = self.server.application  # type: ignore[attr-defined]
        if not isinstance(application, LocalWebApplication):  # pragma: no cover - wiring
            raise TypeError("local_web.application_absent")
        return application

    def log_message(self, *_args: object) -> None:
        """Keep routine local requests out of the console.

        A browser refresh should not produce request noise; this service has no
        separate request-log sink.
        """

    def _respond(self, response: LocalWebResponse, *, set_session: bool) -> None:
        self.close_connection |= cast(_Server, self.server).stopping.is_set()
        try:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header(
                "Content-Security-Policy",
                response.content_security_policy or CONTENT_SECURITY_POLICY,
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cache-Control", response.cache_control or "no-store")
            if response.location is not None:
                self.send_header("Location", response.location)
            if self.close_connection:
                # Said out loud so the client stops reusing a socket this service is
                # about to drop, rather than discovering it on the next request.
                self.send_header("Connection", "close")
            if set_session:
                self.send_header(
                    "Set-Cookie",
                    f"{self.application.cookie_name}={self.application.session_token}; Path=/; "
                    "SameSite=Strict; HttpOnly",
                )
            self.end_headers()
            self.wfile.write(response.body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Delivery can fail after an operation has committed. Close only the
            # socket; never retry dispatch or pretend the client received its result.
            self.close_connection = True
        # Stop can begin after the headers were sent. Finish this response but
        # never read another request from its keep-alive socket afterwards.
        self.close_connection |= cast(_Server, self.server).stopping.is_set()

    def _discard_announced_body(self) -> None:
        """Read and drop a refused request's announced body after its answer (V551).

        A socket closed with unread bytes is reset, and a client under load could read the
        reset before the refusal (WinError 10053). The body is drained in bounded chunks,
        within the admitted bound and the socket's timeout, and never kept; a length the
        preflight refused as malformed or oversized stays unread, as before.
        """
        declared = (self.headers.get("Content-Length") or "0").strip()
        if not declared.isdigit() or int(declared) > MAXIMUM_REQUEST_BODY_BYTES:
            return
        remaining = int(declared)
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    return
                remaining -= len(chunk)
        except OSError:
            return

    def _serve(self, method: str) -> None:
        if cast(_Server, self.server).stopping.is_set():
            self.close_connection = True
            self._respond(
                self.application._refuse(HTTPStatus.SERVICE_UNAVAILABLE, "local_web.stopping"),
                set_session=False,
            )
            self._discard_announced_body()
            return
        headers = {key: value for key, value in self.headers.items()}
        outcome = self.application.preflight(method=method, target=self.path, headers=headers)
        if isinstance(outcome, LocalWebResponse):
            # Refused before the body. Nothing announced is read, so an
            # oversized or hostile request costs one response rather than an
            # allocation -- and the connection has to end, because the bytes the
            # client said it would send are still unread and would otherwise be
            # parsed as the next request on this socket.
            self.close_connection = True
            self._respond(outcome, set_session=False)
            self._discard_announced_body()
            return
        body = self.rfile.read(outcome.content_length) if outcome.content_length else b""
        self._respond(self.application.dispatch(outcome, body), set_session=outcome.issues_session)

    def do_GET(self) -> None:
        self._serve("GET")

    def do_POST(self) -> None:
        self._serve("POST")


class _Bound(Protocol):
    @property
    def server_address(self) -> Any: ...

    def server_close(self) -> None: ...


def bind_browser_safe[B: _Bound](bind: Callable[[], B], *, assigned: bool) -> B:
    """Bind once; when the OS assigned a port a browser refuses, bind again while holding it.

    Each refused socket stays open until a usable one is bound, so the OS cannot hand the
    same port back, and is closed then. A port the caller asked for is kept as asked.

    Args:
        bind: Binds one server and returns it.
        assigned: True when the port was left to the OS (port 0).

    Returns:
        The bound server; after `BROWSER_SAFE_REBINDS` refused ports, the last one.
    """
    refused: list[B] = []
    try:
        while True:
            server = bind()
            if (
                not assigned
                or int(server.server_address[1]) not in BROWSER_REFUSED_PORTS
                or len(refused) >= BROWSER_SAFE_REBINDS
            ):
                return server
            refused.append(server)
    finally:
        for held in refused:
            held.server_close()


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    """A request thread can admit a task, so it is a writer and must be joined.

    Daemon threads are the ones nobody waits for. Here that would mean a request
    still touching the application while the workspace lease is being released,
    which is exactly the second writer the lease exists to prevent.
    """

    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], application: LocalWebApplication) -> None:
        self.application = application
        self.stopping = threading.Event()
        self._handlers: list[threading.Thread] = []
        self._handler_lock = threading.Lock()
        super().__init__(address, _Handler)

    def process_request(self, request: object, client_address: object) -> None:
        """Own the request thread rather than handing it to the base class.

        The stdlib mixin keeps its own private list and joins it inside
        `server_close`. Tracking them here instead means shutdown can say
        *whether* the threads finished, which is the fact a caller holding a
        workspace lease has to act on.
        """
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address),
            name="local-web-request",
            daemon=False,
        )
        with self._handler_lock:
            self._handlers = [value for value in self._handlers if value.is_alive()]
            self._handlers.append(thread)
        thread.start()

    def live_request_threads(self) -> tuple[threading.Thread, ...]:
        with self._handler_lock:
            return tuple(value for value in self._handlers if value.is_alive())

    def join_request_threads(self, timeout: float | None) -> bool:
        """Join every handler. True when none survived."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._handler_lock:
            pending = tuple(self._handlers)
        for thread in pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        return not self.live_request_threads()


@dataclass
class LocalWebService:
    """The socket, the thread and the URL. Started and stopped as a unit."""

    application: LocalWebApplication
    port: int = 0
    _server: _Server | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def bound_port(self) -> int:
        """Return the port bound by the started loopback service."""
        if self._server is None:
            raise LocalWebError("local_web.service_not_started")
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        """Return the started service's loopback URL."""
        return f"http://{LOOPBACK_HOST}:{self.bound_port}/"

    @property
    def launch_url(self) -> str:
        """Return the URL that gives its browser the session, printed and opened once."""
        return f"{self.url.rstrip('/')}{LAUNCH_PATH}?key={self.application.launch_key}"

    def start(self) -> str:
        """Bind loopback, learn the port, and answer on it. Returns the local URL.

        All or nothing. A bind that succeeds and a thread that then fails to
        start would leave a listening socket nobody holds a handle to, so the
        socket is closed on the way out of any failure and the service is left
        exactly as unstarted as it was.
        """
        if self._server is not None:
            return self.url
        server = bind_browser_safe(
            lambda: _Server((LOOPBACK_HOST, self.port), self.application),
            assigned=self.port == 0,
        )
        try:
            self._server = server
            host = f"{LOOPBACK_HOST}:{self.bound_port}"
            # The allow-list is fixed at bind time from the port the OS actually
            # gave us, so `Host` cannot be a wildcard and cannot be widened later.
            self.application.allowed_hosts = frozenset({host, f"localhost:{self.bound_port}"})
            self.application.cookie_name = session_cookie_name(self.bound_port)
            thread = threading.Thread(target=server.serve_forever, name="local-web", daemon=False)
            self._thread = thread
            thread.start()
        except BaseException:
            self._server, self._thread = None, None
            self.application.allowed_hosts = frozenset()
            server.server_close()
            raise
        return self.url

    def stop(self, *, timeout: float | None = None) -> bool:
        """Stop accepting, then join every request thread. True when all ended.

        The order is the whole point. Accepting stops first, so no new handler
        can start behind the shutdown; the accept loop is joined; the listening
        socket is released; and only then are the handlers still mid-request
        waited on. A caller that holds the workspace lease reads the return
        value, because a surviving handler is a surviving writer.
        """
        server, thread = self._server, self._thread
        if server is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        server.stopping.set()
        server.shutdown()
        if thread is not None:
            thread.join(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
        server.server_close()
        joined = server.join_request_threads(
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        if joined:
            self._server, self._thread = None, None
        return joined

    def live_request_threads(self) -> tuple[threading.Thread, ...]:
        """Handler threads still running. Empty is the only clean shutdown."""
        return () if self._server is None else self._server.live_request_threads()

    def __enter__(self) -> LocalWebService:
        """Start the loopback service for the context scope."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop and join the service when leaving the context scope."""
        self.stop()


__all__ = [
    "CONTENT_SECURITY_POLICY",
    "LOOPBACK_HOST",
    "MAXIMUM_REQUEST_BODY_BYTES",
    "SESSION_COOKIE",
    "SESSION_HEADER",
    "AdmittedRequest",
    "LocalWebApplication",
    "LocalWebError",
    "LocalWebResponse",
    "LocalWebService",
    "Route",
    "report_policy",
    "session_cookie_name",
]
