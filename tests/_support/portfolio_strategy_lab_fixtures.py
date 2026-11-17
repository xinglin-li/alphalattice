"""Package fixtures for the Local Web and public Portfolio application suites."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application import client
from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver


@pytest.fixture
def live(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[LocalPortfolioWebSession]:
    """One booted product: session, application, dispatcher and loopback socket. Its workspace
    is its own, by its id too: a Host a wider fixture serves shares this process's machine."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-local-web-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
        **({"clock": request.param} if hasattr(request, "param") else {}),
    )
    session.start()
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Record CLI exchanges while returning the answer selected by the test."""
    state = SimpleNamespace(answer={"status": "OK"}, raw=None, sent=[], workspaces=[])

    class Host:
        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")
            state.workspaces.append(workspace)

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            state.sent.append(dict(document))
            answer = state.answer(document) if callable(state.answer) else state.answer
            raw = state.raw if state.raw is not None else json.dumps(answer).encode("utf-8")
            return answer, raw

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    return state


@pytest.fixture(scope="module")
def read_only_live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalPortfolioWebSession]:
    """Share one synthetic Host for envelope, refusal and timeout reads that admit no Task."""
    workspace = tmp_path_factory.mktemp("cli-read")
    host = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-cli-read-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
    )
    host.start()
    try:
        yield host
        assert host.session is not None and not host.session.task_control_registry.tasks()
    finally:
        host.stop()


@pytest.fixture(scope="module")
def completed_host(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalPortfolioWebSession]:
    """Share one synthetic Host with one completed default Portfolio run. Its workspace has an id
    of its own: a folder made again under the same name is another workspace."""
    from tests.portfolio_strategy_lab.local_web_support import _run_to_completion

    workspace = tmp_path_factory.mktemp("local-web-completed")
    host = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-local-web-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
    )
    host.start()
    try:
        _run_to_completion(host)
        assert host.session is not None
        tasks = host.session.task_control_registry.tasks()
        assert len(tasks) == 1, [(t.task_kind, t.lifecycle, str(t.task_id)) for t in tasks]
        yield host
    finally:
        host.stop()
