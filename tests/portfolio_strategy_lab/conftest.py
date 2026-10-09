"""Package fixtures for the Local Web and public Portfolio application suites."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver


@pytest.fixture(autouse=True)
def reader_zone_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workbench reads an instant in the reader's zone; the Node harnesses these
    tests spawn read it in UTC, so an expected face never depends on the machine's zone."""

    monkeypatch.setenv("TZ", "UTC")


@pytest.fixture
def live(tmp_path: Path) -> Iterator[LocalPortfolioWebSession]:
    """One booted product: session, application, dispatcher and loopback socket."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-local-web"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    try:
        yield session
    finally:
        session.stop()
