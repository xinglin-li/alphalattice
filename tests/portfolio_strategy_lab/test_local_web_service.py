"""Local Web transport, lifecycle and product routes."""

from __future__ import annotations

import http.client
import http.cookiejar
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from http.client import HTTPConnection, HTTPException
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    HANDLED_OPERATION_ROUTES,
    OPERATION_ROUTES,
    LocalPortfolioWebSession,
    LocalWebSessionError,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioRunCommand,
)
from alphalattice.control.task_control.queue import write_queue_setting
from alphalattice.control.task_control.registry import TaskQueueFull
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.interface.local_application.web import (
    BROWSER_REFUSED_PORTS,
    CONTENT_SECURITY_POLICY,
    SESSION_COOKIE,
    SESSION_HEADER,
    LocalWebApplication,
    LocalWebService,
    bind_browser_safe,
    session_cookie_name,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PortfolioResearchTaskAdapter,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    format_book_change,
    format_book_weight,
)
from tests.portfolio_strategy_lab.cli_support import _cli
from tests.portfolio_strategy_lab.local_web_support import (
    _agent,
    _InterruptsOnce,
    _json,
    _manifest,
    _raw,
    _request,
    _resolved,
    _Resolver,
    _run_to_completion,
)


def test_the_service_binds_only_to_loopback(read_only_live: LocalPortfolioWebSession) -> None:
    "No LAN listener, measured by trying to reach it from a routable address."
    port = read_only_live.web.bound_port
    assert read_only_live.url.startswith("http://127.0.0.1:")
    family_host = read_only_live.web._server.server_address[0]
    assert family_host == "127.0.0.1"
    addresses = {
        info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    }
    reachable = []
    for address in addresses - {"127.0.0.1"}:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1.5)
        try:
            probe.connect((address, port))
            reachable.append(address)
        except OSError:
            pass
        finally:
            probe.close()
    assert reachable == [], reachable


def test_every_response_carries_the_local_only_headers(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    status, headers, body = _request(read_only_live, "/")
    assert status == 200
    assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert "Access-Control-Allow-Origin" not in headers
    assert "Set-Cookie" not in headers
    assert "<title>AlphaLattice · Local Web</title>" in body.decode("utf-8")


def test_assets_are_cached_by_content_hash_and_nothing_else_is(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "Assets are cached by content hash and nothing else is."
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/interface/local_application/assets/workbench-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _status, headers, body = _request(read_only_live, "/")
    page = body.decode("utf-8")
    assert headers.get("Cache-Control") == "no-store"
    for name in ("workbench-prelude.js", "workbench.css", "workbench.js"):
        hashed = manifest[name]
        assert hashed in page, (name, hashed)
        status, hashed_headers, hashed_body = _request(read_only_live, hashed)
        assert status == 200
        assert hashed_headers.get("Cache-Control") == "public, max-age=31536000, immutable"
        status, plain_headers, plain_body = _request(read_only_live, f"/{name}")
        assert status == 200
        assert plain_headers.get("Cache-Control") == "no-store"
        assert plain_body == hashed_body
    assert manifest["workbench.zh.js"] not in page
    assert _request(read_only_live, manifest["workbench.zh.js"])[0] == 200
    before_links = page.split("<link", 1)[0]
    assert "workbench-prelude" in before_links, "the prelude runs before the stylesheet"
    assert _request(read_only_live, "/api/session")[1].get("Cache-Control") == "no-store"
    routes = _json(read_only_live, "/api/session")["routes"]
    assert routes == {
        op: {"method": m, "path": p} for m, p, op in (*OPERATION_ROUTES, *HANDLED_OPERATION_ROUTES)
    }
    assert read_only_live.web is not None
    assert len(routes) == len(OPERATION_ROUTES) + len(HANDLED_OPERATION_ROUTES)
    for operation, route in routes.items():
        assert (route["method"], route["path"]) in read_only_live.web.application.routes, operation
    assert routes["EVIDENCE_LEDGER"] == {"method": "GET", "path": "/api/evidence-cro/ledger"}
    assert routes["DATA_UPDATE_RUN"] == {"method": "POST", "path": "/api/data-update/run"}
    assert routes["RESEARCH_UPDATE_AUTOMATION_CONFIGURE"] == {
        "method": "POST",
        "path": "/api/research-update/automation",
    }
    assert {"EVIDENCE_REFRESH", "EVIDENCE_SELECT", "CRO_REVIEW", "EVIDENCE_CRO_EXPORT"} <= set(
        routes
    )


def test_no_asset_reaches_outside_the_machine(read_only_live: LocalPortfolioWebSession) -> None:
    "No CDN, font host, analytics beacon or absolute URL anywhere in the page."
    pages = []
    for path in ("/", "/workbench.css", "/workbench.js"):
        _status, _headers, body = _request(read_only_live, path)
        pages.append(body.decode("utf-8"))
    joined = "\n".join(pages)
    joined = joined.replace("http://www.w3.org/2000/svg", "")
    for marker in ("http://", "https://", "//cdn", "googleapis", "analytics", "fonts."):
        assert marker not in joined, marker


@pytest.mark.parametrize(
    ("label", "kwargs", "expected"),
    [
        ("bad host", {"host": "evil.example"}, "host_not_allowed"),
        ("bad origin", {"origin": "http://evil.example"}, "origin_not_allowed"),
        ("absent origin", {"omit_origin": True}, "origin_absent_on_mutation"),
        (
            "form content type",
            {"content_type": "application/x-www-form-urlencoded"},
            "content_type_not_json",
        ),
        ("absent token", {"token": None}, "session_token_absent"),
        ("absent cookie", {"cookie": None}, "session_token_absent"),
        ("wrong token", {"token": "0" * 43}, "session_token_invalid"),
        ("wrong cookie", {"cookie": "0" * 43}, "session_token_invalid"),
    ],
)
def test_a_mutation_is_refused_before_any_application_work(
    read_only_live: LocalPortfolioWebSession, label: str, kwargs: dict[str, Any], expected: str
) -> None:
    "Eight ways to be unauthorised, and no task admitted by any of them."
    admissions_before = read_only_live.dispatcher.admissions
    del label
    before = len(_json(read_only_live, "/api/tasks")["tasks"])
    status, _headers, body = _request(
        read_only_live, "/api/run", method="POST", payload={}, **kwargs
    )
    assert status == 403, (status, body[:300])
    assert expected in json.loads(body)["refused"]
    assert read_only_live.dispatcher.admissions == admissions_before
    assert len(_json(read_only_live, "/api/tasks")["tasks"]) == before


def test_a_refused_write_ends_its_connection_without_a_reset(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "A refused write ends its connection without a reset."
    body = json.dumps({"padding": "x" * 4096}).encode("utf-8")
    port = read_only_live.web.bound_port
    head = "\r\n".join(
        (
            "POST /api/experiments/handoff HTTP/1.1",
            f"Host: 127.0.0.1:{port}",
            f"Origin: http://127.0.0.1:{port}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            "",
            "",
        )
    ).encode("ascii")
    for _ in range(5):
        with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
            client.sendall(head)
            time.sleep(0.2)
            client.sendall(body)
            received = b""
            while chunk := client.recv(65536):
                received += chunk
        assert received.startswith(b"HTTP/1.1 403"), received[:200]
        assert b"session_token_absent" in received, received[-300:]


def _launch_path(live: LocalPortfolioWebSession) -> str:
    selected = urlsplit(live.launch_url)
    return selected.path + ("?" + selected.query if selected.query else "")


def test_only_the_launch_url_gives_a_browser_the_session(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "Only the launch URL gives a browser the session."
    token = read_only_live.web.application.session_token
    for path in ("/", "/workbench.html"):
        status, headers, _body = _request(read_only_live, path)
        assert status == 200 and "Set-Cookie" not in headers, path
    status, _headers, body = _request(read_only_live, "/api/session", cookie=None)
    refused = json.loads(body)
    assert status == 403 and refused["refused"] == "local_web.session_token_absent"
    assert refused["next_action"] == "REOPEN_FROM_LAUNCH_URL"
    assert token.encode() not in body
    status, _headers, body = _request(read_only_live, "/api/session", cookie="stale")
    assert status == 403 and json.loads(body)["refused"] == "local_web.session_token_invalid"
    for path in ("/launch", "/launch?key=guess", _launch_path(read_only_live) + "&key=again"):
        status, headers, body = _request(read_only_live, path)
        assert status == 403 and "Set-Cookie" not in headers, path
        assert json.loads(body)["refused"] == "local_web.launch_key_invalid"
    status, headers, _body = _request(read_only_live, _launch_path(read_only_live))
    assert status == 303 and headers["Location"] == "/workbench.html"
    assert f"{read_only_live.web.application.cookie_name}={token};" in headers["Set-Cookie"]
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    status, _headers, body = _request(read_only_live, "/api/session", cookie="valid")
    assert status == 200 and json.loads(body)["session_token"] == token


def test_browser_session_cookies_are_port_scoped_and_restart_replaces_only_its_own():
    "behavior: shared localhost cookie jars keep concurrent Hosts independent by port."

    def application(label):
        web = LocalWebApplication()
        web.add_asset("/workbench.html", label.encode(), "text/html")

        @web.route("GET", "/api/session", session=True)
        def session(_query, _payload):
            return {"session_token": web.session_token, "label": label}

        @web.route("POST", "/api/write", mutates=True)
        def write(_query, _payload):
            return {"written_by": label}

        return web

    def open_url(opener, url):
        request = urllib.request.Request(url)
        try:
            with opener.open(request, timeout=5) as response:
                body = response.read()
                try:
                    value = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    value = body.decode("utf-8", "replace")
                return (response.status, value)
        except urllib.error.HTTPError as error:
            return (error.code, json.loads(error.read() or b"{}"))

    def write_url(opener, url, token):
        origin = url.split("/api/", 1)[0]
        request = urllib.request.Request(
            url,
            data=b"{}",
            method="POST",
            headers={SESSION_HEADER: token, "Origin": origin, "Content-Type": "application/json"},
        )
        try:
            with opener.open(request, timeout=5) as response:
                return (response.status, json.loads(response.read() or b"{}"))
        except urllib.error.HTTPError as error:
            return (error.code, json.loads(error.read() or b"{}"))

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
    )
    first = LocalWebService(application("first"))
    second = LocalWebService(application("second"))
    services = [first, second]
    try:
        first_url = first.start().rstrip("/")
        second_url = second.start().rstrip("/")
        first_port, second_port = (first.bound_port, second.bound_port)
        assert first_port != second_port
        for service, base in ((first, first_url), (second, second_url)):
            status, _ = open_url(opener, f"{base}/launch?key={service.application.launch_key}")
            assert status == 200
        first_cookie = session_cookie_name(first_port)
        second_cookie = session_cookie_name(second_port)
        assert first_cookie != second_cookie
        assert {cookie.name for cookie in jar} >= {first_cookie, second_cookie}
        first_status, first_session = open_url(opener, f"{first_url}/api/session")
        second_status, second_session = open_url(opener, f"{second_url}/api/session")
        first_token = first.application.session_token
        second_token = second.application.session_token
        assert first_status == second_status == 200
        assert first_session["session_token"] == first_token
        assert second_session["session_token"] == second_token
        assert write_url(opener, f"{first_url}/api/write", first_token)[0] == 200
        assert write_url(opener, f"{second_url}/api/write", second_token)[0] == 200
        status, refused = write_url(opener, f"{first_url}/api/write", second_token)
        assert status == 403 and refused["refused"] == "local_web.session_token_invalid"
        first.stop()
        replacement = LocalWebService(application("replacement"), port=first_port)
        services.append(replacement)
        replacement_url = replacement.start().rstrip("/")
        assert replacement.bound_port == first_port
        old_token = first_token
        new_token = replacement.application.session_token
        assert new_token != old_token
        status, _ = open_url(
            opener, f"{replacement_url}/launch?key={replacement.application.launch_key}"
        )
        assert status == 200
        assert open_url(opener, f"{replacement_url}/api/session")[1]["session_token"] == new_token
        old_cookie_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        old_cookie_request = urllib.request.Request(f"{replacement_url}/api/session")
        old_cookie_request.add_header("Cookie", f"{first_cookie}={old_token}")
        with pytest.raises(urllib.error.HTTPError) as error:
            old_cookie_opener.open(old_cookie_request, timeout=5)
        assert error.value.code == 403
        assert open_url(opener, f"{second_url}/api/session")[1]["session_token"] == second_token
        assert write_url(opener, f"{replacement_url}/api/write", new_token)[0] == 200
    finally:
        for service in reversed(services):
            service.stop(timeout=5)


def test_an_unknown_route_is_not_a_filesystem_path(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "No route opens a path, a store or a database, including by traversal."
    for path in (
        "/../pyproject.toml",
        "/etc/passwd",
        "/runtime/artifacts",
        "/api/../app.js%00",
        "/api/store",
    ):
        status, _headers, _body = _request(read_only_live, path)
        assert status in {403, 404}, (path, status)


@pytest.mark.parametrize("prepared_input", [None, "b" * 64])
def test_a_session_portfolio_replan_keeps_its_optional_input_choice(
    read_only_live: LocalPortfolioWebSession, monkeypatch: pytest.MonkeyPatch, prepared_input
) -> None:
    "A session portfolio replan keeps its optional input choice."
    admissions_before = read_only_live.dispatcher.admissions
    task_rows_before = _json(read_only_live, "/api/tasks")["tasks"]
    assert read_only_live.operations is not None and read_only_live.operations.updates is not None
    owner = read_only_live.operations.updates
    original = owner.plan
    seen = []

    def plan(package_id, input_hash, observed_through):
        seen.append((package_id, input_hash, observed_through))
        return original(package_id, input_hash, observed_through)

    monkeypatch.setattr(owner, "plan", plan)
    route = _json(read_only_live, "/api/session")["routes"]["PORTFOLIO_UPDATE_PLAN"]
    payload = {
        "strategy_package_id": "synthetic-uninstalled",
        "prepared_input_hash": prepared_input,
        "observed_through": "2026-09-30",
    }
    answer = _json(read_only_live, route["path"], method=route["method"], payload=payload)
    assert answer["status"] == "REFUSED", answer
    assert seen == [("synthetic-uninstalled", prepared_input, date(2026, 9, 30))]
    for invalid in (
        {**payload, "strategy_package_id": None},
        {**payload, "observed_through": None},
        {**payload, "prepared_input_hash": 1},
        {**payload, "unknown": "value"},
    ):
        status, _headers, body = _request(
            read_only_live, route["path"], method=route["method"], payload=invalid
        )
        assert status == 400, body
        assert json.loads(body)["refused"] == "portfolio_update.plan_fields_invalid"
    assert len(seen) == 1
    assert (
        read_only_live.dispatcher is not None
        and read_only_live.dispatcher.admissions == admissions_before
    )
    assert _json(read_only_live, "/api/tasks")["tasks"] == task_rows_before


def test_run_admission_returns_before_the_work_and_the_task_progresses(
    live: LocalPortfolioWebSession, monkeypatch
) -> None:
    "The whole point of the dispatcher: an answer now, a task that moves after."
    published, release = (threading.Event(), threading.Event())
    execute = PortfolioRunCommand.execute

    def hold_after_publication(command, task_id):
        execute(command, task_id)
        published.set()
        assert release.wait(10), "test did not release the completed command"

    monkeypatch.setattr(PortfolioRunCommand, "execute", hold_after_publication)
    started = time.perf_counter()
    admitted = _json(live, "/api/run", method="POST", payload={})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert admitted["disposition"] == "ADMITTED"
    assert admitted["task_id"]
    assert admitted["lifecycle"] in {"QUEUED", "RUNNING"}
    assert elapsed_ms < 2000.0, elapsed_ms
    try:
        assert published.wait(10)
        assert _json(live, "/api/results")["results"]
        held = _json(live, f"/api/status?task_id={admitted['task_id']}")
        assert held["lifecycle"] == "RUNNING"
        assert held["task_id"] == admitted["task_id"]
    finally:
        release.set()
    live.dispatcher.drain_for_tests()
    final = _json(live, f"/api/status?task_id={admitted['task_id']}")
    assert final["lifecycle"] == "SUCCEEDED"
    assert final["verified_stage_count"] == final["total_stage_count"]
    assert final["worker_failure"] is None


def test_capacity_is_the_workspaces_own_and_is_reported_truthfully(
    live: LocalPortfolioWebSession,
) -> None:
    "Capacity is the workspace's own and is reported truthfully."
    application = live.application
    dispatcher = live.dispatcher
    assert application is not None and dispatcher is not None
    write_queue_setting(
        live.workspace / "runtime", 1, chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )
    submissions = [
        dispatcher.submit(
            PortfolioRunCommand(
                application=application, spec=PortfolioResearchSpec.create(top_k=20 + index)
            )
        )
        for index in range(4)
    ]
    dispositions = [value.disposition for value in submissions]
    assert dispositions[0] == "ADMITTED"
    assert "REFUSED_QUEUE_FULL" in dispositions, dispositions
    assert dispositions.index("REFUSED_QUEUE_FULL") <= 2, dispositions
    for value in submissions:
        if value.disposition == "REFUSED_QUEUE_FULL":
            assert value.task_id is None
            assert value.refusal_detail
        else:
            assert value.task_id is not None
            assert value.lifecycle in {"QUEUED", "RUNNING"}
    assert dispatcher.admissions == dispositions.count("ADMITTED")
    dispatcher.drain_for_tests()


def test_a_full_queue_refuses_a_run_before_its_plan(
    live: LocalPortfolioWebSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    "the RUN resolved its sources, scope and caches, then learned the"
    application = live.application
    assert application is not None and live.operations is not None
    registry = live.session.task_control_registry

    def full() -> None:
        raise TaskQueueFull("task_control.queue_full: 1 Tasks wait in the queue's 1 places")

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a full queue planned the run")

    monkeypatch.setattr(registry, "check_capacity", full)
    monkeypatch.setattr(application, "plan", never)
    answer = live.operations.run({})
    assert answer["disposition"] == "REFUSED_QUEUE_FULL" and answer["task_id"] is None
    assert "queue_full" in str(answer["refusal_detail"])


@pytest.mark.parametrize("unexpected_error", (False, True))
def test_cancellation_is_requested_not_enacted(
    live: LocalPortfolioWebSession, monkeypatch, unexpected_error
) -> None:
    "The dispatcher forwards; the runner decides at its own stage boundary."
    from alphalattice.control.task_control.runner import TaskControlRunner

    run_next = TaskControlRunner.run_next
    cancellations = []

    def cancel_before_claim(runner, *, expected_task_id=None, expected_task_hash=None):
        cancellations.append(
            _json(live, "/api/cancel", method="POST", payload={"task_id": str(expected_task_id)})
        )
        if unexpected_error:
            raise ValueError("unrelated_worker_defect")
        return run_next(
            runner, expected_task_id=expected_task_id, expected_task_hash=expected_task_hash
        )

    monkeypatch.setattr(TaskControlRunner, "run_next", cancel_before_claim)
    admitted = _json(live, "/api/run", method="POST", payload={})
    live.dispatcher.drain_for_tests()
    assert len(cancellations) == 1 and cancellations[0]["cancel_requested"] is True
    assert cancellations[0]["task_id"] == admitted["task_id"]
    final = _json(live, f"/api/status?task_id={admitted['task_id']}")
    assert final["lifecycle"] == "CANCELLED"
    assert final["worker_failure"] == (
        "ValueError: unrelated_worker_defect" if unexpected_error else None
    )


def test_a_failed_start_leaves_no_lease_worker_or_socket(tmp_path: Path) -> None:
    "A failed start leaves no lease worker or socket."
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken = int(blocker.getsockname()[1])
    before = {thread.name for thread in threading.enumerate()}
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-failed-start"),
        resolver=_Resolver(_resolved()),
        port=taken,
    )
    try:
        with pytest.raises(OSError):
            session.start()
        assert session.session is None
        assert session.dispatcher is None
        assert session.web is None
        assert {thread.name for thread in threading.enumerate()} - before == set()
        second = LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=_manifest("qa-failed-start"),
            resolver=_Resolver(_resolved()),
        )
        second.start()
        try:
            assert _json(second, "/api/session")["workspace_id"] == "qa-failed-start"
        finally:
            second.stop()
    finally:
        blocker.close()


def test_shutdown_joins_a_slow_authenticated_handler(live: LocalPortfolioWebSession) -> None:
    "A request thread is a writer, so stopping waits for it rather than past it."
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def _slow(_query: object, _payload: object) -> object:
        entered.set()
        release.wait(timeout=20.0)
        order.append("handler")
        return {"slow": True}

    live.web.application.route("GET", "/api/qa-slow")(_slow)
    caller = threading.Thread(target=lambda: _request(live, "/api/qa-slow"), daemon=True)
    caller.start()
    assert entered.wait(timeout=20.0)

    def _stop() -> None:
        live.stop()
        order.append("stop")

    stopper = threading.Thread(target=_stop, daemon=True)
    stopper.start()
    time.sleep(0.4)
    assert stopper.is_alive()
    assert live.session is not None
    release.set()
    stopper.join(timeout=30.0)
    caller.join(timeout=30.0)
    assert not stopper.is_alive()
    assert order == ["handler", "stop"], order
    assert live.web is None
    assert live.session is None


def test_a_bounded_stop_will_not_release_the_lease_while_a_command_runs(tmp_path: Path) -> None:
    "The old bounded join returned with the worker alive. Now it refuses to."
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-bounded"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    running = threading.Event()
    release = threading.Event()

    class _SlowCommand:
        command_kind = "qa_slow_command"

        def admit(self) -> CommandAdmission:
            return CommandAdmission(task_id=uuid4(), lifecycle="QUEUED")

        def execute(self, task_id: UUID) -> None:
            del task_id
            running.set()
            release.wait(timeout=60.0)

    try:
        session.dispatcher.submit(_SlowCommand())
        assert running.wait(timeout=20.0)
        with pytest.raises(LocalWebSessionError):
            session.stop(timeout=0.05)
        assert session.session is not None
        assert session.dispatcher is not None
        assert session.dispatcher.worker_alive
        contender = LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=_manifest("qa-bounded"),
            resolver=_Resolver(_resolved()),
        )
        with pytest.raises(RuntimeError, match="writer is already owned"):
            contender.start()
        release.set()
        session.stop()
        assert session.session is None
    finally:
        release.set()
        with suppress(Exception):
            session.stop()
    successor = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-bounded"),
        resolver=_Resolver(_resolved()),
    )
    successor.start()
    try:
        assert _json(successor, "/api/session")["workspace_id"] == "qa-bounded"
    finally:
        successor.stop()


def test_shutdown_drains_current_request_without_admitting_keepalive_followups():
    from alphalattice.interface.local_application.web import LocalWebApplication, LocalWebService

    app = LocalWebApplication()
    entered, release = (threading.Event(), threading.Event())
    calls = []

    @app.route("GET", "/api/session")
    def read(_query, _body):
        calls.append("read")
        entered.set()
        assert release.wait(5)
        return {"status": "READ"}

    service = LocalWebService(app)
    service.start()
    connection = HTTPConnection("127.0.0.1", service.bound_port, timeout=3)
    try:
        connection.request("GET", "/api/session")
        assert entered.wait(3)
        assert not service.stop(timeout=0.05)
        release.set()
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        with pytest.raises((OSError, HTTPException)):
            connection.request("GET", "/api/session")
            connection.getresponse().read()
        assert calls == ["read"]
    finally:
        release.set()
        connection.close()
        assert service.stop(timeout=5)


@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
def test_disconnected_response_closes_transport_without_retrying_or_masking_application_errors(
    failure, monkeypatch
):
    from alphalattice.interface.local_application.web import (
        AdmittedRequest,
        LocalWebApplication,
        LocalWebResponse,
        _Handler,
    )

    application = LocalWebApplication()
    response = LocalWebResponse(200, b"committed result", "application/json")
    admitted = AdmittedRequest(None, None, {}, 0, False)
    calls = []
    monkeypatch.setattr(application, "preflight", lambda **_: admitted)

    def dispatch(*_args):
        calls.append("dispatch")
        return response

    monkeypatch.setattr(application, "dispatch", dispatch)
    handler = object.__new__(_Handler)
    handler.server = SimpleNamespace(application=application, stopping=threading.Event())
    handler.close_connection = False
    handler.path, handler.headers, handler.rfile = ("/", {}, BytesIO())
    handler.send_response = lambda *_: None
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None

    def disconnected(_body):
        calls.append("write")
        raise failure("closed transport")

    handler.wfile = SimpleNamespace(write=disconnected)
    handler._serve("GET")
    assert calls == ["dispatch", "write"] and handler.close_connection
    monkeypatch.setattr(
        application, "dispatch", lambda *_: (_ for _ in ()).throw(ValueError("owner failure"))
    )
    with pytest.raises(ValueError, match="owner failure"):
        handler._serve("GET")


def _handcrafted(session: LocalPortfolioWebSession, *, token: str, length: str) -> bytes:
    port = session.web.bound_port
    lines = (
        "POST /api/run HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        f"Origin: http://127.0.0.1:{port}",
        "Content-Type: application/json",
        f"{SESSION_HEADER}: {token}",
        f"Cookie: {SESSION_COOKIE}={token}",
        f"Content-Length: {length}",
        "",
        "",
    )
    return "\r\n".join(lines).encode("ascii")


def test_an_oversized_body_is_refused_before_it_is_read(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "The bound is checked against the declared length, not the bytes received."
    task_rows_before = _json(read_only_live, "/api/tasks")["tasks"]
    token = read_only_live.web.application.session_token
    started = time.perf_counter()
    status, payload = _raw(
        read_only_live, _handcrafted(read_only_live, token=token, length=str(200 * 1024 * 1024))
    )
    elapsed = time.perf_counter() - started
    assert status == 413, payload[:300]
    assert b"local_web.body_too_large" in payload
    assert b"Connection: close" in payload
    assert elapsed < 4.0
    assert _json(read_only_live, "/api/tasks")["tasks"] == task_rows_before


def test_an_unauthorised_oversized_request_does_no_application_work(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "Refused on authorisation -- before the length, the body and the handler."
    task_rows_before = _json(read_only_live, "/api/tasks")["tasks"]
    result_rows_before = _json(read_only_live, "/api/results")["results"]
    application = read_only_live.web.application
    served_before = application.served
    status, payload = _raw(
        read_only_live, _handcrafted(read_only_live, token="not-the-token", length="104857600")
    )
    assert status == 403, payload[:300]
    assert b"local_web.session_token_invalid" in payload
    assert b"Connection: close" in payload
    assert _json(read_only_live, "/api/tasks")["tasks"] == task_rows_before
    assert _json(read_only_live, "/api/results")["results"] == result_rows_before
    assert application.served == served_before + 2
    deadline = time.monotonic() + 10.0
    while read_only_live.web.live_request_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert read_only_live.web.live_request_threads() == ()


@pytest.mark.parametrize("declared", ["-5", "abc", "12 34", "1.5"])
def test_a_malformed_content_length_is_refused_with_a_stable_reason(
    read_only_live: LocalPortfolioWebSession, declared: str
) -> None:
    "Negative, fractional and non-numeric all fail the same named way."
    task_rows_before = _json(read_only_live, "/api/tasks")["tasks"]
    token = read_only_live.web.application.session_token
    status, payload = _raw(
        read_only_live, _handcrafted(read_only_live, token=token, length=declared)
    )
    assert status == 400, payload[:300]
    assert b"local_web.content_length_invalid" in payload
    assert _json(read_only_live, "/api/tasks")["tasks"] == task_rows_before


@pytest.mark.parametrize("phase", ("WORK", "RESULT_PUBLISHED", "CANCEL"))
def test_local_web_reopens_after_forced_process_death(tmp_path: Path, phase: str) -> None:
    from tests.portfolio_strategy_lab.process_death import forced_process

    workspace = tmp_path / "workspace"
    with forced_process(workspace, __name__, phase) as interrupted:
        pass
    resolver = _Resolver(_resolved())
    service = LocalPortfolioWebSession(
        workspace=workspace, workspace_manifest=_manifest("qa-hard-stop"), resolver=resolver
    )
    service.start()
    try:
        service.dispatcher.drain_for_tests()
        task_id = interrupted["task_id"]
        listed = _json(service, "/api/tasks")["tasks"]
        assert len(listed) == 1
        assert listed[0]["lifecycle"] == ("CANCELLED" if phase == "CANCEL" else "SUCCEEDED")
        status = _json(service, f"/api/status?task_id={task_id}")
        task = service.session.task_control_registry.task(UUID(task_id))
        assert task.input.input_hash == interrupted["input_hash"]
        assert len(service.session.task_control_registry.tasks()) == 1
        results = _json(service, "/api/results")["results"]
        if phase == "CANCEL":
            assert status["lifecycle"] == "CANCELLED" and results == []
            assert service.resumed_task_ids == () and resolver.numerical_calls == 0
        else:
            assert service.resumed_task_ids == (UUID(task_id),)
            assert status["lifecycle"] == "SUCCEEDED" and len(results) == 1
            result_hash = results[0]["result_hash"]
            assert _json(service, f"/api/report?result_hash={result_hash}")
            assert _json(service, f"/api/export?result_hash={result_hash}")
            if phase == "RESULT_PUBLISHED":
                assert result_hash == interrupted["result_hash"] and resolver.numerical_calls == 0
    finally:
        service.stop()


def test_process_cleanup_does_not_adopt_children_of_reused_parent_ids() -> None:
    from tests.portfolio_strategy_lab.process_death import _creation_filtered_tree

    parents = {20: 10, 30: 20, 40: 30, 50: 20}
    births = {10: 100, 20: 101, 30: 50, 40: 105, 50: 102}
    assert _creation_filtered_tree(10, parents, births.get) == (10, 20, 50)
    births[20] = 90
    assert _creation_filtered_tree(10, parents, births.get) == (10,)


def test_export_round_trips_from_workspace_id_and_a_spec_path(
    completed_host: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    "The manifest is enough to ask the same question again, and nothing more."
    result_hash = _run_to_completion(completed_host)
    manifest = _json(completed_host, f"/api/export?result_hash={result_hash}")
    assert manifest["workspace_id"] == "qa-local-web"
    assert manifest["result_hash"] == result_hash
    assert (
        manifest["command"] == "alphalattice --workspace <dir> strategy-book run --file <spec.json>"
    )
    for banned in ("--root", "--result", "--report-hash", "--program-hash"):
        assert banned not in manifest["command"], banned
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(manifest["spec"]), encoding="utf-8")
    reloaded = json.loads(spec_path.read_text(encoding="utf-8"))
    replayed = _json(completed_host, "/api/plan", method="POST", payload={"spec": reloaded})
    assert replayed["spec_hash"] == manifest["spec_hash"]
    assert replayed["exact_cache_hit"] is True
    assert replayed["cached_result_hash"] == result_hash


def test_export_describes_the_result_not_whatever_is_in_the_form(
    completed_host: LocalPortfolioWebSession,
) -> None:
    "A later plan must not change what an earlier result exports."
    result_hash = _run_to_completion(completed_host)
    exported = _json(completed_host, f"/api/export?result_hash={result_hash}")
    moved_on = _json(completed_host, "/api/plan", method="POST", payload={"spec": {"top_k": 22}})
    assert moved_on["spec_hash"] != exported["spec_hash"]
    again = _json(completed_host, f"/api/export?result_hash={result_hash}")
    assert again["spec_hash"] == exported["spec_hash"]
    assert again["spec"] == exported["spec"]
    assert again["result_hash"] == result_hash
    assert again["spec"]["top_k"] != 22


def test_an_unfrozen_candidate_reports_not_frozen(read_only_live: LocalPortfolioWebSession) -> None:
    status = _json(read_only_live, f"/api/finalization?candidate_hash={'a' * 64}")
    assert status["disposition"] == "NOT_FROZEN"
    assert status["candidate"] is None


def test_development_replay_copy_makes_no_forward_claim(
    completed_host: LocalPortfolioWebSession,
) -> None:
    "No `today`, no `current holdings`, no `intended trades`, anywhere a user reads."
    result_hash = _run_to_completion(completed_host)
    surfaces = [
        _request(completed_host, "/")[2].decode("utf-8"),
        _request(completed_host, "/workbench.js")[2].decode("utf-8"),
        _request(completed_host, f"/report?result_hash={result_hash}")[2].decode("utf-8"),
    ]
    session = _json(completed_host, "/api/session")
    assert session["execution_mode"] == "DEVELOPMENT_REPLAY"
    for surface in surfaces:
        lowered = surface.lower()
        for claim in (
            "today",
            "current holdings",
            "intended trade",
            "will trade",
            "recommended trade",
        ):
            assert claim not in lowered, claim


def test_compare_names_differences_without_selecting(live: LocalPortfolioWebSession) -> None:
    first = _run_to_completion(live)
    admitted = _json(
        live,
        "/api/run",
        method="POST",
        payload={"spec": {**_default_spec_document(live), "cost_bps_per_side": "10"}},
    )
    assert admitted["disposition"] == "ADMITTED"
    live.dispatcher.drain_for_tests()
    results = [item["result_hash"] for item in _json(live, "/api/results")["results"]]
    second = next(value for value in results if value != first)
    body = _json(live, f"/api/compare?left={first}&right={second}")
    assert body == _agent(
        live,
        PortfolioResearchAgentRequest(
            operation="COMPARE", left_result_hash=first, right_result_hash=second
        ),
    )
    assert body["disposition"] == "DECLARED_PATH_COMPARISON_NO_SELECTION"
    assert "cost_bps_per_side" in body["differing_controls"]
    assert body["shares_execution_ledger"] is True
    assert [dimension["dimension"] for dimension in body["dimensions"]] == [
        "HOLDINGS",
        "CONCENTRATION",
        "COST",
        "TURNOVER",
        "PERFORMANCE",
        "RISK",
        "LIMITATIONS",
    ]
    by_dimension = {item["dimension"]: item["metrics"] for item in body["dimensions"]}
    assert {item["label"] for item in by_dimension["HOLDINGS"]} >= {
        "Names held",
        "Absolute weight change",
    }
    assert {item["label"] for item in by_dimension["CONCENTRATION"]} >= {"Effective N"}
    assert {item["label"] for item in by_dimension["COST"]} >= {"Cost per side"}
    assert {item["label"] for item in by_dimension["TURNOVER"]} >= {"Mean one-way turnover"}
    assert {item["label"] for item in by_dimension["PERFORMANCE"]} >= {"Cumulative net wealth"}
    assert {item["label"] for item in by_dimension["RISK"]} >= {"Predicted volatility"}
    assert {item["label"] for item in by_dimension["LIMITATIONS"]} == {"Declared limitations"}
    encoded = json.dumps(body).lower()
    for word in ("winner", "ranked", "preferred", "recommend", "better", "score_rank"):
        assert word not in encoded, word
    assert body["kind"] == "INSTALLED_RESULT_COMPARISON"
    assert set(body) == {
        "kind",
        "disposition",
        "shares_execution_ledger",
        "differing_controls",
        "left",
        "right",
        "dimensions",
    }


def test_plan_leads_with_strategy_work_limits_and_lawful_actions(
    live: LocalPortfolioWebSession,
) -> None:
    plan = _json(live, "/api/plan", method="POST", payload={})
    assert plan["strategy"]["strategy_package_id"] == plan["strategy_package_id"]
    assert plan["strategy"]["package_hash"] == plan["strategy_package_hash"]
    assert plan["available_interval"] == {
        "start": plan["coverage"]["common_watermark_start"],
        "end": plan["coverage"]["common_watermark_end"],
        "sessions": plan["coverage"]["common_session_count"],
        "selected_start": plan["coverage"]["common_watermark_start"],
        "selected_end": plan["coverage"]["common_watermark_end"],
    }
    assert plan["cache"]["state"] == "FULL_NUMERICAL_MISS"
    assert plan["estimated_work"]["basis"] == "UPPER_BOUND_OVER_CANDIDATE_SUPPORT"
    assert plan["estimated_work"]["optimizer_calls"] == 0
    assert plan["limitations"]["capacity"] == "NOT_MODELED"
    assert plan["limitations"]["refused_capability_count"] == len(live.service.controls.refusals)
    assert plan["next_lawful_actions"][0] == "ADMIT_DECLARED_PATH_RUN"
    assert {control["control_id"] for control in plan["available_controls"]} == {
        control.control_id for control in live.service.controls.controls
    }
    exit_rank = next(
        control for control in plan["available_controls"] if control["control_id"] == "exit_rank"
    )
    assert exit_rank["min"] == str(PortfolioResearchSpec.default().top_k)
    assert exit_rank["max"]


def _default_spec_document(live: LocalPortfolioWebSession) -> dict[str, Any]:
    manifest_spec = PortfolioResearchSpec.default()
    return {
        "top_k": manifest_spec.top_k,
        "tranches": manifest_spec.tranches,
        "exit_rank": manifest_spec.exit_rank,
        "weight_rule": manifest_spec.weight_rule,
        "cost_bps_per_side": str(manifest_spec.cost.cost_bps_per_side),
        "secondary_benchmark_view": manifest_spec.secondary_benchmark_view,
        "report_unit": manifest_spec.report_unit,
        "study_start": None,
        "study_end": None,
    }


def test_every_registered_route_answers(live: LocalPortfolioWebSession) -> None:
    "Every registered route answers."
    result_hash = _run_to_completion(live)
    frozen = _json(live, "/api/freeze", method="POST", payload={"result_hash": result_hash})
    application = live.web.application
    query = {
        "/api/status": f"?task_id={_json(live, '/api/tasks')['tasks'][0]['task_id']}",
        "/api/report": f"?result_hash={result_hash}",
        "/api/export": f"?result_hash={result_hash}",
        "/report": f"?result_hash={result_hash}",
        "/api/compare": f"?left={result_hash}&right={result_hash}",
        "/api/finalization": f"?candidate_hash={frozen['candidate_hash']}",
    }
    answered: dict[str, int] = {}
    for method, path in sorted(application.routes):
        if method != "GET":
            continue
        if path.startswith("/api/client/"):
            from alphalattice.interface.local_application.client import LocalResearchClient

            client = LocalResearchClient(live.workspace)
            client_body = client.activity() if path == "/api/client/activity" else client.request()
            assert client_body["workspace_id"] == live.workspace_manifest.workspace_id
            assert "refused" not in client_body
            answered[path] = 200
            continue
        status, _headers, body = _request(live, f"{path}{query.get(path, '')}")
        answered[path] = status
        assert status in {200, 400}, (path, status, body[:200])
        if status == 400:
            assert "handler_failed" not in json.loads(body)["refused"], path
    assert answered["/api/controls"] == 200
    assert answered["/api/session"] == 200
    assert answered["/api/results"] == 200
    assert answered["/api/report"] == 200
    assert answered["/api/finalization"] == 200
    for path in ("/", "/workbench.html", "/workbench.css", "/workbench.js"):
        assert _request(live, path)[0] == 200, path


def test_the_control_catalog_projection_matches_the_installed_catalog(
    read_only_live: LocalPortfolioWebSession,
) -> None:
    "The HTTP catalog carries the installed catalog's identity and ordered control ids."
    body = _json(read_only_live, "/api/controls")
    catalog = read_only_live.service.controls
    assert body["catalog_hash"] == catalog.catalog_hash
    assert [item["control_id"] for item in body["controls"]] == [
        item.control_id for item in catalog.controls
    ]
    assert [item["control_id"] for item in body["refused"]] == [
        item.control_id for item in catalog.refusals
    ]


def _crash_child(workspace: str, phase: str) -> None:
    resolver = _Resolver(_resolved())
    service = LocalPortfolioWebSession(
        workspace=Path(workspace), workspace_manifest=_manifest("qa-hard-stop"), resolver=resolver
    )
    resolve = resolver.resolve
    verify = PortfolioResearchTaskAdapter.verify_stage

    def stop(result_hash=None):
        registry = service.session.task_control_registry
        task = registry.active_task()
        assert task is not None
        if phase == "CANCEL":
            task, _ = registry.request_cancel(
                task_id=task.task_id,
                expected_task_hash=task.record_hash,
                observed_at=service.clock(),
            )
        print(
            "CRASH_BOUNDARY:"
            + json.dumps(
                {
                    "pid": os.getpid(),
                    "task_id": str(task.task_id),
                    "record_hash": task.record_hash,
                    "input_hash": task.input.input_hash,
                    "result_hash": result_hash,
                }
            ),
            flush=True,
        )
        threading.Event().wait()

    def resolve_or_stop(**kwargs):
        if phase in {"WORK", "CANCEL"}:
            stop()
        return resolve(**kwargs)

    def verify_or_stop(self, **kwargs):
        if phase == "RESULT_PUBLISHED":
            stop(kwargs["evidence"][0].content_hash)
        return verify(self, **kwargs)

    resolver.resolve = resolve_or_stop
    PortfolioResearchTaskAdapter.verify_stage = verify_or_stop
    service.start()
    _json(service, "/api/run", method="POST", payload={})
    service.dispatcher.drain_for_tests()
    service.stop()


def test_a_published_input_refreshes_every_manifest_holder(live: LocalPortfolioWebSession) -> None:
    "A published input refreshes every manifest holder."
    operations = live.operations
    assert operations is not None and operations.automation is not None
    applications = (
        operations.experiments,
        operations.data_update,
        operations.scoring,
        operations.calibration,
        operations.updates,
    )
    assert all(application is not None for application in applications)
    current = SimpleNamespace(manifest_hash="b" * 64)
    operations._hold(current)
    assert operations.workspace_manifest is current
    assert all(application.manifest is current for application in applications)
    assert operations.automation.manifest_hash == current.manifest_hash


@pytest.mark.untyped_failure
def test_a_genuine_fault_stays_a_fault_at_every_web_owner_entry(
    read_only_live, monkeypatch
) -> None:
    "each public handler dependency and the activation owner still"
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_untyped_fault_matrix

    exercise_untyped_fault_matrix(read_only_live, monkeypatch)


def test_a_port_the_browser_refuses_is_never_the_workbenchs():
    "Windows gave a port-0 bind 1720, which Chromium refuses"

    class Bound:
        def __init__(self, port: int) -> None:
            self.server_address = ("127.0.0.1", port)
            self.closed = False

        def server_close(self) -> None:
            self.closed = True

    ports = iter((1720, 6000, 49152))
    made: list[Bound] = []

    def bind() -> Bound:
        made.append(Bound(next(ports)))
        return made[-1]

    kept = bind_browser_safe(bind, assigned=True)
    assert kept.server_address[1] == 49152 and (not kept.closed)
    assert [(server.server_address[1], server.closed) for server in made[:-1]] == [
        (1720, True),
        (6000, True),
    ]
    assert {1720, 5060, 6000, 10080} <= BROWSER_REFUSED_PORTS
    asked = bind_browser_safe(lambda: Bound(1720), assigned=False)
    assert asked.server_address[1] == 1720 and (not asked.closed)


def test_human_and_agent_share_plan_task_status_report_and_freeze(
    live: LocalPortfolioWebSession,
) -> None:
    spec: dict[str, object] = {"top_k": 22, "exit_rank": 45, "cost_bps_per_side": "13"}
    selected = spec
    resolver = live.resolver
    assert isinstance(resolver, _Resolver)
    before_numerical = resolver.numerical_calls
    assert resolver.numerical_calls == 0
    browser_plan = _json(live, "/api/plan", method="POST", payload={"spec": spec})
    agent_plan = _agent(live, PortfolioResearchAgentRequest(operation="PLAN", spec=spec))
    assert agent_plan == browser_plan
    plan = browser_plan
    assert plan["optimizer_call_count"] == 0
    assert resolver.numerical_calls == before_numerical
    assert _json(live, "/api/tasks")["tasks"] == []
    assert live.dispatcher.admissions == 0
    assert plan["legal_recovery"]
    request = dict(plan["next_requests"]["run"])
    assert request["operation"] == "RUN"
    assert all((str(request["spec"][key]) == str(value) for key, value in selected.items()))
    assert request["spec"]["strategy_package_id"] == plan["strategy_package_id"]
    assert request["spec"]["score_source_mode"] == plan["score_source_mode"]
    assert request.pop("operation") == "RUN"
    admitted = _agent(live, PortfolioResearchAgentRequest(operation="RUN", spec=spec))
    assert admitted["disposition"] == "ADMITTED" and admitted["task_id"], admitted
    assert live.dispatcher.admissions == 1
    task_id = UUID(str(admitted["task_id"]))
    live.dispatcher.drain_for_tests()
    agent_status = _agent(live, PortfolioResearchAgentRequest(operation="STATUS", task_id=task_id))
    browser_status = _json(live, f"/api/status?task_id={task_id}")
    assert agent_status == browser_status
    assert agent_status["lifecycle"] == "SUCCEEDED"
    agent_results = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS"))
    assert agent_results == _json(live, "/api/results")
    assert len(agent_results["results"]) == 1
    result_hash = str(agent_results["results"][0]["result_hash"])
    report = _agent(
        live, PortfolioResearchAgentRequest(operation="REPORT", result_hash=result_hash)
    )
    assert report == _json(live, f"/api/report?result_hash={result_hash}")
    assert report["review_selector"] == {"result_hash": result_hash}
    assert report["next_requests"]["evidence_preview"] == {
        "operation": "EVIDENCE_PREVIEW",
        "result_hash": result_hash,
    }
    assert {request["operation"] for request in report["next_requests"].values()} == {
        "EVIDENCE_CRO",
        "EVIDENCE_PREVIEW",
    }
    assert report["next_requests"]["review"] == {
        "operation": "EVIDENCE_CRO",
        "result_hash": result_hash,
    }
    resolver = live.resolver
    assert isinstance(resolver, _Resolver)
    before = resolver.numerical_calls
    admissions = live.dispatcher.admissions
    tasks = _json(live, "/api/tasks")
    plan = _json(live, "/api/plan", method="POST", payload={"spec": spec})
    assert plan["exact_cache_hit"] is True
    assert plan["estimated_score_replays"] == 0
    assert plan["estimated_risk_surface_builds"] == 0
    request = dict(plan["next_requests"]["run"])
    assert request.pop("operation") == "RUN"
    route = _json(live, "/api/session")["routes"]["RUN"]
    reused = _json(live, route["path"], method=route["method"], payload=request)
    assert reused["disposition"] == "REUSED_EXACT"
    assert reused["result_hash"] == result_hash
    assert reused["task_id"] is None
    assert resolver.numerical_calls == before
    assert live.dispatcher.admissions == admissions
    assert _json(live, "/api/tasks") == tasks
    assert resolver.numerical_calls - before == 0
    frozen = _agent(
        live, PortfolioResearchAgentRequest(operation="FREEZE", result_hash=result_hash)
    )
    assert frozen["development_result_hash"] == result_hash
    assert frozen["development_task_id"]
    assert frozen["development_run_hash"]
    assert frozen["control_receipt_hash"]
    assert frozen["pre_protected_state_hash"]
    frozen_status = _json(live, f"/api/finalization?candidate_hash={frozen['candidate_hash']}")
    assert frozen_status["disposition"] == "AWAITING_PROTECTED_AUTHORITY"
    assert frozen_status["next_lawful_action"] == "WAIT_FOR_PROTECTED_AUTHORITY"
    assert frozen_status["stage_11_action_available"] is False
    assert frozen_status["handoff_hash"] is None
    assert frozen_status["released_result_hash"] is None
    assert "protected" in frozen_status["detail"]
    candidate_hash = str(frozen["candidate_hash"])
    agent_finalization = _agent(
        live, PortfolioResearchAgentRequest(operation="FINALIZATION", candidate_hash=candidate_hash)
    )
    assert agent_finalization == _json(live, f"/api/finalization?candidate_hash={candidate_hash}")
    assert agent_finalization["disposition"] == "AWAITING_PROTECTED_AUTHORITY"
    assert agent_finalization["stage_11_action_available"] is False
    assert _agent(
        live, PortfolioResearchAgentRequest(operation="EVIDENCE_CRO", result_hash=result_hash)
    ) == _json(live, f"/api/evidence-cro?result_hash={result_hash}")
    spec = {"top_k": 500}
    before = _agent(live, PortfolioResearchAgentRequest(operation="TASKS"))
    status, _headers, body = _request(live, "/api/plan", method="POST", payload={"spec": spec})
    assert status == 400
    browser_refusal = json.loads(body)
    agent_refusal = _agent(live, PortfolioResearchAgentRequest(operation="PLAN", spec=spec))
    assert agent_refusal == browser_refusal
    assert "TOP_K_OUTSIDE_ADMITTED_RANGE" in agent_refusal["refused"]
    assert _agent(live, PortfolioResearchAgentRequest(operation="TASKS")) == before


@pytest.mark.parametrize(
    "association", ["indexed", "legacy", "redirected", "missing-keys", "wrong-shape"]
)
def test_results_selects_reused_task_metadata_without_scanning_reports(
    live: LocalPortfolioWebSession, monkeypatch, association: str
) -> None:
    "the selected Task's sealed association is exact, and damage is a named refusal."
    from alphalattice.control.product_host.publication.portfolio_research import (
        PortfolioResearchPipelineManifest,
    )
    from alphalattice.interface.local_application.answers import answer_problem

    assert live.application is not None and live.operations is not None
    pipeline = live.application.pipeline
    from alphalattice.control.task_control.contracts import TaskEvidence, TaskStageReceipt
    from alphalattice.investment.portfolio_strategy_lab.application.task import (
        PortfolioResearchTaskAdapter,
    )

    application = live.application
    registry = live.session.task_control_registry
    admitted = application.admit(spec=PortfolioResearchSpec.create())
    task = registry.task(admitted.task_id)
    adapter = PortfolioResearchTaskAdapter(
        workspace_id=application.workspace_id,
        spec=admitted.spec,
        program=admitted.program,
        resolved=admitted.resolve_once,
        coverage=admitted.authorities.coverage,
        authorities_hash=admitted.authorities.authorities_hash,
        admission_hash=admitted.planned.preview.admission_hash,
        workspace_manifest_hash=admitted.planned.preview.workspace_manifest_hash,
        strategy_catalog_hash=application.resolver.strategy_catalog_hash,
        selected_strategy_package_id=admitted.authorities.strategy_package_id,
        executor=application.executor,
    )
    started = registry.start_next(
        compatibility=adapter.compatibility(task),
        worker_instance_id=uuid4(),
        observed_at=task.admitted_at,
        expected_task_id=task.task_id,
    )
    if started is None:
        raise RuntimeError("metadata fixture could not claim its admitted Task")
    task, execution = started
    (definition,) = task.plan.work_items
    item = registry.begin_work_item(
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id=definition.stage_id,
        observed_at=task.admitted_at,
    )
    evidence = (
        TaskEvidence(
            evidence_kind=definition.required_evidence_kinds[0],
            reference=f"fixture://portfolio-result/{'c' * 64}",
            content_hash="c" * 64,
        ),
    )
    registry.mark_ready(
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id=definition.stage_id,
        evidence=evidence,
        observed_at=task.admitted_at,
    )
    completed = registry.verify_work_item(
        TaskStageReceipt.from_identity(
            receipt_id=uuid4(),
            task_id=task.task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            work_item_definition_hash=item.definition_hash,
            verifier_id=definition.verifier_id,
            evidence=evidence,
            status="VERIFIED",
            failure_code=None,
            observed_at=task.admitted_at,
        )
    )
    first = completed.task_id
    reused = uuid4()
    moment = datetime(2026, 10, 2, tzinfo=UTC)
    values = [
        PortfolioResearchPipelineManifest.create(
            workspace_id=live.operations.workspace_manifest.workspace_id,
            task_id=task,
            task_record_hash=completed.record_hash if task == first else str(index) * 64,
            program_hash="b" * 64,
            result_hash="c" * 64,
            report_hash="d" * 64,
            completed_at=moment + timedelta(minutes=index),
        )
        for index, task in enumerate((first, reused), start=1)
    ]
    for value in values:
        pipeline.publish(value)

    def no_report(*_args, **_kwargs):
        raise AssertionError("Result metadata opened a report")

    monkeypatch.setattr(type(live.operations), "report", no_report)
    collection = _json(live, "/api/results")
    assert next(row for row in collection["results"] if row["result_hash"] == "c" * 64)[
        "task_id"
    ] == str(first)
    index_path = pipeline._task_index(reused)
    if association == "legacy":
        index_path.unlink()
    elif association == "redirected":
        index_path.write_text(
            json.dumps({"task_id": str(reused), "manifest_hash": values[0].manifest_hash}),
            encoding="utf-8",
        )
    elif association in {"missing-keys", "wrong-shape"}:
        index_path.write_text("{}" if association == "missing-keys" else "[]", encoding="utf-8")
    pipeline._result_index("e" * 64).write_text("{", encoding="utf-8")
    browser = _json(live, f"/api/results?task_id={reused}")
    agent = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS", task_id=reused))
    assert agent == browser
    assert answer_problem("RESULTS", browser, {"task_id": str(reused)}) is None
    if association in {"indexed", "legacy"}:
        (row,) = browser["results"]
        assert row == {
            "result_hash": values[1].result_hash,
            "report_hash": values[1].report_hash,
            "program_hash": values[1].program_hash,
            "task_id": str(reused),
            "completed_at": values[1].completed_at.isoformat(),
        }
        assert not browser.get("refusals")
    else:
        assert browser["results"] == []
        (refusal,) = browser["refusals"]
        assert refusal["status"] == "REFUSED" and refusal["task_id"] == str(reused)
        assert refusal["record_id"] == f"task:{reused}"
        assert refusal["next_requests"]["task"] == {"operation": "STATUS", "task_id": str(reused)}
        assert refusal["detail"] and "handler_failed" not in refusal["failure_code"]
        if association == "redirected":
            assert refusal["failure_code"] == "portfolio_application.pipeline_task_index_tampered"
    healthy_task_id = str(first)
    healthy_result = {"result_hash": values[0].result_hash}
    bad_hash = "e" * 64
    pipeline._result_index(bad_hash).write_text("{", encoding="utf-8")
    agent_results = _agent(live, PortfolioResearchAgentRequest(operation="RESULTS"))
    assert agent_results == _json(live, "/api/results")
    assert [row["result_hash"] for row in agent_results["results"]] == [
        healthy_result["result_hash"]
    ]
    (result_refusal,) = agent_results["refusals"]
    assert result_refusal["result_hash"] == bad_hash
    assert result_refusal["record_id"] == bad_hash
    assert result_refusal["entry_id"] == f"result:{bad_hash}"
    assert "task_id" not in result_refusal
    agent_history = _agent(
        live, PortfolioResearchAgentRequest(operation="RESEARCH_HISTORY", history_limit=50)
    )
    browser_history = _json(live, "/api/research-history?history_limit=50")
    assert agent_history == browser_history
    assert agent_history["status"] == "PARTIAL_METADATA_READBACK"
    (healthy_history,) = [
        row
        for row in agent_history["entries"]
        if row["entry_id"] == f"result:{healthy_result['result_hash']}"
    ]
    assert healthy_history["task_id"] == healthy_task_id
    assert healthy_history["status"] == "SUCCEEDED"
    blocked = next(
        row for row in agent_history["blocked_entries"] if row["result_hash"] == bad_hash
    )
    assert blocked["status"] == "REFUSED"
    assert blocked["kind"] == "INSTALLED_RESULT"
    assert blocked["entry_id"] == f"result:{bad_hash}"
    assert "task_id" not in blocked
    status, _headers, encoded = _request(live, "/api/results?task_id=not-a-task")
    body = json.loads(encoded)
    assert status == 400
    assert body["failure_code"] == "local_web.task_id_invalid"
    assert body["next_action"] == "COPY_FULL_TASK_ID_FROM_HISTORY"


def test_an_idle_keep_alive_connection_does_not_block_shutdown(
    tmp_path: Path, read_only_live, completed_host
) -> None:
    "An idle keep alive Host stops all its threads beside other running Hosts."
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-keepalive"),
        resolver=_Resolver(_resolved()),
    )
    before = set(threading.enumerate())
    session.start()
    port = session.web.bound_port
    idle = socket.create_connection(("127.0.0.1", port), timeout=10.0)
    try:
        idle.sendall(
            "\r\n".join(
                (
                    "GET /api/session HTTP/1.1",
                    f"Host: 127.0.0.1:{port}",
                    f"Cookie: {SESSION_COOKIE}={session.web.application.session_token}",
                    "Connection: keep-alive",
                    "",
                    "",
                )
            ).encode("ascii")
        )
        assert idle.recv(4096).startswith(b"HTTP/1.1 200")
        started = time.perf_counter()
        session.stop()
        elapsed = time.perf_counter() - started
    finally:
        idle.close()
        with suppress(Exception):
            session.stop()
    assert elapsed < 30.0, elapsed
    assert session.session is None
    remaining = {
        thread.ident: thread.name for thread in threading.enumerate() if thread not in before
    }
    assert remaining == {}, remaining


def test_every_web_handler_preserves_every_typed_owner_exception(
    read_only_live, monkeypatch
) -> None:
    "Every web handler preserves every typed owner exception."
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_registered_handler_matrix

    exercise_registered_handler_matrix(read_only_live, monkeypatch, typed_owner_cases())
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_activation_owner_matrix

    exercise_activation_owner_matrix(read_only_live, monkeypatch, typed_owner_cases())
    from tests.portfolio_strategy_lab.typed_owner_refusals import typed_owner_cases
    from tests.portfolio_strategy_lab.web_refusal_support import exercise_observed_owner_matrix

    exercise_observed_owner_matrix(read_only_live, monkeypatch, typed_owner_cases())


def test_report_html_is_the_sealed_page_read_back(completed_host: LocalPortfolioWebSession) -> None:
    "Immediately inspectable, and not re-rendered by the service."
    result_hash = _run_to_completion(completed_host)
    status, headers, body = _request(completed_host, f"/report?result_hash={result_hash}")
    page = body.decode("utf-8")
    sealed = completed_host.service.open_html(result_hash)
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert page == sealed
    policy = headers["Content-Security-Policy"]
    assert "style-src 'self' 'sha256-" in policy
    assert "unsafe-inline" not in policy
    assert policy.startswith("default-src 'none'")
    assert "connect-src 'self'" in policy


def test_every_visible_number_equals_a_typed_fact(completed_host: LocalPortfolioWebSession) -> None:
    "The browser renders values; it does not produce them."
    result_hash = _run_to_completion(completed_host)
    body = _json(completed_host, f"/api/report?result_hash={result_hash}")
    report = completed_host.service.report(result_hash)
    readouts = completed_host.service.readouts(result_hash)
    assert body["readouts"]["distinct_names_held"] == report.window_end_distinct_names
    assert body["readouts"]["effective_n"] == report.window_end_effective_n
    assert body["readouts"]["cumulative_net_wealth"] == report.window_cumulative_net_wealth
    assert body["readouts"]["cost_bps_per_side"] == readouts.cost_bps_per_side
    assert body["readouts"]["cost_bps_round_trip"] == readouts.cost_bps_round_trip
    assert body["window"]["selected_session_count"] == report.window_guard.selected_session_count
    assert [tuple(v) for v in body["controls"]] == list(report.control_receipt.selected)
    book = body["book"]
    sealed = completed_host.service.report(result_hash)
    assert book["formation_session"] == sealed.window_end_book.formation_session.isoformat()
    assert book["held_count"] == sealed.window_end_book.held_count
    assert book["held_count"] == body["readouts"]["distinct_names_held"]
    assert len(book["positions"]) == len(sealed.window_end_book.positions)
    assert book["change_boundary"] == sealed.window_end_book.change_boundary
    assert book["change_boundary"] in {
        "PRECEDING_FORMATION",
        "SEALED_CONTINUATION_BOUNDARY",
        "FLAT_PATH_OPENING",
    }
    assert book["preceding_formation_session"] == (
        None
        if sealed.window_end_book.preceding_formation_session is None
        else sealed.window_end_book.preceding_formation_session.isoformat()
    )
    for projected, position in zip(
        book["positions"], sealed.window_end_book.positions, strict=True
    ):
        assert projected["listing_id"] == position.listing_id
        assert projected["weight"] == format_book_weight(position.weight)
        assert projected["weight_change_bp"] == format_book_change(position.weight_change)
        assert projected["disposition"] == position.disposition
    assert book["absolute_weight_change_total"] == format_book_weight(
        sealed.window_end_book.absolute_weight_change_total
    )


def test_the_evidence_and_cro_section_names_why_it_waits(
    completed_host: LocalPortfolioWebSession,
) -> None:
    "The Host's book review names its typed missing-authority state and refusal."
    result_hash = _run_to_completion(completed_host)
    body = _json(completed_host, f"/api/evidence-cro?result_hash={result_hash}")
    assert body["state"] == "EVIDENCE_AUTHORITY_NOT_ADMITTED"
    preview = _json(completed_host, f"/api/evidence/preview?result_hash={result_hash}")
    assert preview["failure_code"] == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY", preview


def test_a_route_that_writes_takes_the_write_check(live: LocalPortfolioWebSession) -> None:
    "A route that writes takes the write check."
    for path in [
        "/api/experiments/training-inputs/prepare",
        "/api/workspace/storage/plan",
        "/api/workspace/storage/cap",
        "/api/research-inputs/plan",
        "/api/experiments/training-inputs/plan",
        "/api/research-strategies/plan",
        "/api/workspace/data-issues/preview",
        "/api/workspace/preparation/plan",
        "/api/experiments/plan",
        "/api/experiments/handoff",
        "/api/experiments/foundations/preview",
    ]:
        status, _headers, body = _request(live, path, method="POST", payload={}, token=None)
        assert status == 403, (status, body[:300])
        assert "session_token_absent" in json.loads(body)["refused"]
        if path == "/api/workspace/storage/cap":
            cap = str(20 * 1024**3)
            changed = _json(
                live,
                "/api/workspace/storage/cap",
                method="POST",
                payload={"storage_cap_bytes": cap},
            )
            assert changed["status"] == "CONFIGURED"
            assert changed["capacity"]["cap_bytes"] == int(cap)
            status, _headers, body = _request(
                live,
                "/api/workspace/storage/cap",
                method="POST",
                payload={"storage_cap_bytes": "0"},
            )
            assert (
                status == 200 and json.loads(body)["failure_code"] == "storage.cap_setting_invalid"
            )
            assert _json(live, "/api/workspace/storage/cap")["capacity"]["cap_bytes"] == int(cap)
            human = _json(
                live,
                "/api/workspace/storage/cap",
                method="POST",
                payload={"storage_cap_bytes": str(25 * 1024**3)},
            )
            assert human["capacity"]["setting"]["chosen_by"] == "HUMAN"
            code, shown, _ = _cli(live.workspace, "storage", "cap")
            assert code == 0 and shown["data"]["capacity"]["cap_bytes"] == 25 * 1024**3


@pytest.mark.parametrize(
    ("never_started", "external_recovery"),
    [(False, False), (False, True), (False, "http"), (True, False)],
)
def test_a_service_restart_recovers_the_same_task_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    never_started: bool,
    external_recovery: bool | str,
) -> None:
    "A genuinely interrupted task is finished by the next service, once."
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    interrupting = _InterruptsOnce(_Resolver(_resolved()))
    first = LocalPortfolioWebSession(
        workspace=workspace, workspace_manifest=_manifest("qa-restart"), resolver=interrupting
    )
    first.start()
    try:
        with monkeypatch.context() as patch:
            if never_started:
                patch.setattr(first.dispatcher._work, "put", lambda item: None)
            admitted = _json(first, "/api/run", method="POST", payload={})
            task_id = admitted["task_id"]
            if not never_started:
                first.dispatcher.drain_for_tests()
        interrupted = _json(first, f"/api/status?task_id={task_id}")
        tasks_before = _json(first, "/api/tasks")["tasks"]
        results_before = _json(first, "/api/results")["results"]
    finally:
        first.stop()
    assert interrupting.failures == (0 if never_started else 1)
    assert interrupted["lifecycle"] == ("QUEUED" if never_started else "RECOVERY_REQUIRED"), (
        interrupted
    )
    assert len(tasks_before) == 1
    assert results_before == []
    second = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-restart"),
        resolver=_Resolver(_resolved()),
    )
    if external_recovery:
        with monkeypatch.context() as patch:
            patch.setattr(LocalPortfolioWebSession, "resume", lambda *a, **k: ())
            second.start()
        second.operations.recover_task = second.resume
    else:
        second.start()
    try:
        if external_recovery:
            from alphalattice.interface.local_application.client import LocalResearchClient

            recovered = (
                _json(second, "/api/recover", method="POST", payload={"task_id": task_id})
                if external_recovery == "http"
                else LocalResearchClient(workspace).request(
                    {"operation": "RECOVER", "task_id": task_id}
                )
            )
            resumed = tuple(UUID(v) for v in recovered["resumed_task_ids"])
        else:
            resumed = second.resumed_task_ids
        second.dispatcher.drain_for_tests()
        status = _json(second, f"/api/status?task_id={task_id}")
        tasks = _json(second, "/api/tasks")["tasks"]
        results = _json(second, "/api/results")["results"]
    finally:
        second.stop()
    assert [str(value) for value in resumed] == [task_id]
    assert status["task_id"] == task_id
    assert status["lifecycle"] == "SUCCEEDED", status
    assert [row["task_id"] for row in tasks] == [task_id]
    assert len(results) == 1


def test_history_and_discovery_keep_peers_without_advancing_unreadable_authority(
    unreadable_task_records,
) -> None:
    "Discovery retains readable history and quarantines unreadable Tasks without continuations."
    data = unreadable_task_records
    live, damaged, peer = (data.live, data.damaged, data.peer)
    before, answers, healthy_result = (data.before, data.answers, data.healthy_result)
    history = answers["RESEARCH_HISTORY"]
    assert any(row["task_id"] == healthy_result for row in history["entries"])
    assert not any(row["task_id"] == str(damaged.task_id) for row in history["entries"])
    exact = _json(live, f"/api/research-history?history_entry_id=task:{damaged.task_id}")
    assert exact["task_id"] == str(damaged.task_id) and exact["status"] == "REFUSED"
    assert exact["next_requests"]["backups"] == {"operation": "WORKSPACE_BACKUPS"}
    assert any(row["task_id"] == str(peer.task_id) for row in answers["TASK_GUARDIAN"]["tasks"])
    assert answers["EXPERIMENTS"]["experiments"] == before["EXPERIMENTS"]["experiments"]
    assert answers["DATA_ISSUES"]["next_requests"] == {
        "pending": {"operation": "PENDING_DECISIONS"}
    }
    assert answers["DATA_ISSUES"]["continuations"] == []
    assert answers["DATA_ISSUES"]["delegations"] == []
    assert answers["DATA_ISSUES"]["task_refusals"]
    assert any(
        row.get("task_id") == str(damaged.task_id) and row["kind"] == "TASK_RECORD_UNREADABLE"
        for row in answers["PENDING_DECISIONS"]["decisions"]
    )


def test_task_pages_keep_session_refusals_and_admission_cursors(unreadable_task_records) -> None:
    "Task pages isolate session refusals while retaining the original admission cursors."
    data = unreadable_task_records
    live, damaged, peer = (data.live, data.damaged, data.peer)
    agent_session, other_session = (data.agent_session, data.other_session)
    before, answers, other_before = (data.before, data.answers, data.other_before)
    session_tasks = answers["SESSION_TASKS"]
    assert session_tasks["tasks"] == [before["SESSION_TASKS"]["tasks"][0]]
    assert session_tasks["next_cursor"] is None
    (session_refusal,) = session_tasks["refusals"]
    assert session_refusal["task_id"] == str(damaged.task_id)
    assert "attention" not in session_refusal
    first_page = _json(live, f"/api/tasks?agent_session={agent_session}&history_limit=1")
    assert first_page["tasks"] == session_tasks["tasks"]
    assert first_page["next_cursor"] == str(peer.task_id)
    assert not first_page.get("refusals")
    last_page = _json(
        live,
        f"/api/tasks?agent_session={agent_session}&history_limit=1&history_cursor={first_page['next_cursor']}",
    )
    assert last_page == {"tasks": [], "next_cursor": None, "refusals": [session_refusal]}
    assert _json(live, f"/api/tasks?agent_session={other_session}") == other_before


def test_task_record_collections_keep_readable_rows_and_offer_named_authority_routes(
    unreadable_task_records,
) -> None:
    "Collections retain readable peers and name unreadable authority without projecting it."
    data = unreadable_task_records
    damaged, peer, alien, answers = (data.damaged, data.peer, data.alien, data.answers)

    def named_refusals(value):
        if isinstance(value, dict):
            if value.get("task_id") == str(damaged.task_id) and value.get("status") == "REFUSED":
                yield value
            for item in value.values():
                yield from named_refusals(item)
        elif isinstance(value, list):
            for item in value:
                yield from named_refusals(item)

    for operation, answer in answers.items():
        refusals = list(named_refusals(answer))
        assert refusals, (operation, answer)
        for refusal in refusals:
            assert refusal["failure_code"] == "task_control.database_authority_unreadable"
            assert not {
                "lifecycle",
                "task_kind",
                "progress",
                "artifact_refs",
                "task_record_hash",
            }.intersection(refusal), refusal
            assert refusal["detail"] and refusal["next_action"]
            assert refusal["next_requests"]["workspace"] == {"operation": "WORKSPACE_SHOW"}
            assert refusal["next_requests"]["backups"] == {"operation": "WORKSPACE_BACKUPS"}
        assert str(alien.task_id) not in json.dumps(answer)
    assert any(row["task_id"] == str(peer.task_id) for row in answers["TASKS"]["tasks"])


def test_unreadable_task_authority_refuses_recovery_and_writes(unreadable_task_records) -> None:
    "Unreadable canonical authority refuses recovery and writes without changing stored records."
    import duckdb

    from alphalattice.control.task_control.registry import TaskRecordAuthorityError
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        TASK_KIND as REVIEW_TASK_KIND,
    )

    data = unreadable_task_records
    live, registry, damaged = (data.live, data.registry, data.damaged)
    answers, projection_before = (data.answers, data.projection_before)
    stored_before, projections_before = (data.stored_before, data.projections_before)
    assert registry.projection_collection((damaged.task_id,)) == projection_before
    assert "acknowledge" not in answers["UPGRADE_OVERVIEW"]["next_requests"]
    assert live.review is not None
    assert REVIEW_TASK_KIND in live.review.recovery_commands(owed_only=False)
    assert REVIEW_TASK_KIND in live.recoverable_task_kinds()
    with pytest.raises(TaskRecordAuthorityError):
        live.review.recovery_commands()
    with pytest.raises(TaskRecordAuthorityError):
        live.resume()
    assert _json(live, "/api/session?context=1")["workspace_id"] == live.workspace_id
    _agent(live, PortfolioResearchAgentRequest(operation="WORKSPACE_SHOW"))
    _json(live, "/api/workspace/backup")
    acknowledgement = live.workspace / "runtime" / "upgrade-overview.json"
    acknowledgement_before = acknowledgement.read_bytes() if acknowledgement.exists() else None
    _status, _headers, body = _request(
        live,
        "/api/upgrade/acknowledge",
        method="POST",
        payload={"upgrade_set_hash": answers["UPGRADE_OVERVIEW"]["installed"]["set_hash"]},
    )
    refused = json.loads(body)
    assert refused["failure_code"] == "task_control.database_authority_unreadable"
    assert (
        acknowledgement.read_bytes() if acknowledgement.exists() else None
    ) == acknowledgement_before
    with pytest.raises(TaskRecordAuthorityError):
        registry.tasks()
    with pytest.raises(TaskRecordAuthorityError):
        live.operations.supervisor.supervise_once()
    with duckdb.connect(str(registry.database_path), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
            ).fetchall()
            == stored_before
        )
        assert (
            connection.execute(
                "SELECT task_id, projection_json FROM workspace_task_projection ORDER BY task_id"
            ).fetchall()
            == projections_before
        )


@pytest.fixture(params=["bad-json", "different-canonical-id"])
def unreadable_task_records(live: LocalPortfolioWebSession, request):
    """Build separate canonical corruption and readable peers for each collection requirement."""
    damage = request.param
    import duckdb

    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        TaskRecord,
    )
    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        RequestProvenance,
    )
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        TASK_KIND as REVIEW_TASK_KIND,
    )
    from tests.workspace_task_runner.task_control_support import task_contract

    assert live.operations is not None
    assert live.session is not None
    registry = live.session.task_control_registry
    write_queue_setting(
        live.workspace / "runtime", "4", chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )
    agent_session = "task-record-collection"
    other_session = "task-record-other-session"
    tasks = []
    for salt, session in (
        ("unreadable-canonical-record", agent_session),
        ("readable-queued-peer", agent_session),
        ("readable-other-session-peer", other_session),
    ):
        envelope, goal, plan = task_contract(salt=salt)
        if salt == "readable-queued-peer":
            envelope = TaskInputEnvelope.create(
                task_kind=REVIEW_TASK_KIND,
                input_schema_id="task-record-collection.review-fixture",
                payload={"prepared_submission": {}, "salt": salt},
            )
            goal = ResearchGoal.create(
                goal_kind="RUN_PORTFOLIO_REVIEW",
                input_hash=envelope.input_hash,
                deliverable_kind="PortfolioReview",
                summary="A queued review command retained beside unreadable authority.",
            )
            plan = ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash=plan.workflow_definition_hash,
                verifier_catalog_hash=plan.verifier_catalog_hash,
                work_items=plan.work_items,
            )
        provenance = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session=session))
        try:
            tasks.append(
                registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
                ).record
            )
        finally:
            REQUEST_PROVENANCE.reset(provenance)
    damaged, peer, other_peer = tasks
    healthy_result = str(peer.task_id)
    damaged, _command = registry.request_cancel(
        task_id=damaged.task_id,
        expected_task_hash=damaged.record_hash,
        observed_at=datetime.now(UTC),
    )
    reads = {
        "RESEARCH_HISTORY": "/api/research-history?history_limit=50",
        "EXPERIMENTS": "/api/experiments",
        "PENDING_DECISIONS": "/api/decisions",
        "TASK_GUARDIAN": "/api/tasks/guardian",
        "UPGRADE_OVERVIEW": "/api/upgrade",
        "DATA_ISSUES": "/api/workspace/data-issues",
        "TASKS": "/api/tasks",
        "SESSION_TASKS": f"/api/tasks?agent_session={agent_session}",
    }
    before = {operation: _json(live, path) for operation, path in reads.items()}
    assert any(row["task_id"] == healthy_result for row in before["RESEARCH_HISTORY"]["entries"])
    assert any(row["task_id"] == str(peer.task_id) for row in before["TASK_GUARDIAN"]["tasks"])
    assert [row["task_id"] for row in before["SESSION_TASKS"]["tasks"]] == [
        str(peer.task_id),
        str(damaged.task_id),
    ]
    other_before = _json(live, f"/api/tasks?agent_session={other_session}")
    assert [row["task_id"] for row in other_before["tasks"]] == [str(other_peer.task_id)]
    projection_before = registry.projection_collection((damaged.task_id,))
    alien = TaskRecord.from_identity(
        **{**damaged.model_dump(exclude={"record_hash"}), "task_id": uuid4()}
    )
    replacement = "{" if damage == "bad-json" else alien.model_dump_json()
    with duckdb.connect(str(registry.database_path)) as connection:
        connection.execute(
            "UPDATE workspace_task SET record_json = ? WHERE task_id = ?",
            [replacement, str(damaged.task_id)],
        )
        stored_before = connection.execute(
            "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
        ).fetchall()
        projections_before = connection.execute(
            "SELECT task_id, projection_json FROM workspace_task_projection ORDER BY task_id"
        ).fetchall()
    answers = {operation: _json(live, path) for operation, path in reads.items()}
    return SimpleNamespace(
        live=live,
        registry=registry,
        damaged=damaged,
        peer=peer,
        alien=alien,
        agent_session=agent_session,
        other_session=other_session,
        before=before,
        other_before=other_before,
        answers=answers,
        projection_before=projection_before,
        stored_before=stored_before,
        projections_before=projections_before,
        healthy_result=healthy_result,
    )
