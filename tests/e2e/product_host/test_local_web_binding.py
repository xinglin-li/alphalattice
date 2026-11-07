"""The local service answers on no routable address, which only a real network probe shows."""

from __future__ import annotations

import socket

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession


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
