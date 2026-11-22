import io
from types import SimpleNamespace
from typing import cast

import pytest

from alphalattice.interface.local_application import web
from alphalattice.interface.local_application.thin_cli import CLI_OPERATION, OPERATIONS_PATH

TEST_SOURCE = "src/alphalattice/interface/local_application/web.py"


class _Trickle(io.BytesIO):
    """A client that sends one byte per receive, then goes silent past the time it is given."""

    def __init__(self, sent: int) -> None:
        super().__init__(b"x" * sent)

    def read1(self, size: int = -1) -> bytes:
        chunk = super().read1(1)
        if not chunk:
            raise TimeoutError("timed out")
        return chunk


@pytest.mark.proves("local_application.a_refused_body_drain_ends_at_its_deadline")
def test_a_trickled_refused_body_is_drained_no_longer_than_its_deadline(monkeypatch):
    """Each read of a refused request's body waits no longer than the drain's time left; a client
    still sending at the deadline, or silent past it, has its connection closed."""
    clock = iter([0.0, 1.0, 2.0, web.DRAIN_SECONDS + 1.0])
    monkeypatch.setattr(web.time, "monotonic", lambda: next(clock))
    timeouts: list[float | None] = []
    connection = SimpleNamespace(gettimeout=lambda: 10.0, settimeout=timeouts.append)
    handler = SimpleNamespace(
        headers={"Content-Length": "4096"}, rfile=_Trickle(4096), connection=connection
    )
    handler.close_connection = False
    web._Handler._discard_announced_body(cast("web._Handler", handler))
    assert handler.close_connection is True and handler.rfile.tell() == 2
    assert timeouts == [web.DRAIN_SECONDS - 1.0, web.DRAIN_SECONDS - 2.0, 10.0]  # then restored
    clock = iter([0.0, 1.0])
    silent = SimpleNamespace(headers={"Content-Length": "8"}, rfile=_Trickle(0))
    silent.connection, silent.close_connection = connection, False
    web._Handler._discard_announced_body(cast("web._Handler", silent))
    assert silent.close_connection is True


@pytest.mark.proves("local_application.cli_request_keeps_read_admission")
@pytest.mark.parametrize(
    ("path", "operation", "read"),
    [
        pytest.param(OPERATIONS_PATH, "TASKS", True, id="ordinary-read"),
        pytest.param(OPERATIONS_PATH, CLI_OPERATION, True, id="bounded-cli-read"),
        pytest.param(OPERATIONS_PATH, "RUN", False, id="ordinary-write"),
        pytest.param("/other", CLI_OPERATION, False, id="carrier-on-another-route"),
    ],
)
def test_bounded_cli_requests_share_the_existing_read_admission(path, operation, read):
    """Only the carrier at its own operation door receives the ordinary read policy."""
    route = web.Route(
        "POST", path, lambda _query, _payload: None, mutates=True, external_client=True
    )
    assert route.admits_read(operation, {"TASKS"}) is read
