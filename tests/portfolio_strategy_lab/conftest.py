"""Package fixtures for the Local Web and public Portfolio application suites."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver


@pytest.fixture(autouse=True)
def reader_zone_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workbench reads an instant in the reader's zone (law 133); the Node harnesses these
    tests spawn read it in UTC, so an expected face never depends on the machine's zone."""

    monkeypatch.setenv("TZ", "UTC")


@pytest.fixture(scope="session")
def workbench_build() -> None:
    """The workbench's built assets (git-ignored), built once per source state for every test
    that serves or reads them. Windows refuses to replace an asset another process holds open,
    and not only xdist workers run at once: the gate runs several pytest processes, and one's
    build failed on `workbench.js` while another's Host served it. So the first process to
    claim the current source's build builds it; every other process, and every later run while
    the source is unchanged, finds the assets current and replaces nothing."""

    from scripts.build_local_web_ui import ASSETS, MANIFEST, ROOT, build

    digest = hashlib.sha256(str(ASSETS).encode())
    inputs = [*(ASSETS / "workbench-source").rglob("*"), *(ASSETS / "fonts").rglob("*")]
    for path in sorted([*inputs, ROOT / "scripts/build_local_web_ui.py"]):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    marks = Path(tempfile.gettempdir()) / "alphalattice-workbench-build"
    marks.mkdir(exist_ok=True)
    done = marks / f"{digest.hexdigest()[:32]}.done"
    claim = done.with_suffix(".claim")

    def current() -> bool:
        # the manifest is written last and names every output by its content hash
        manifest = ASSETS / MANIFEST
        return done.is_file() and manifest.is_file() and manifest.read_bytes() == done.read_bytes()

    for _ in range(1200):  # at most ten minutes for another process's build
        if current():
            return
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            with contextlib.suppress(FileNotFoundError):
                if time.time() - claim.stat().st_mtime > 600:
                    claim.unlink()  # its builder died without releasing it
            time.sleep(0.5)
            continue
        try:
            build(product=True)
            done.write_bytes((ASSETS / MANIFEST).read_bytes())
        finally:
            claim.unlink(missing_ok=True)
        return
    raise TimeoutError(f"the workbench build claimed at {claim} never finished")


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
