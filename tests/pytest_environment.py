"""Own pytest's temporary roots and the QA launcher's confined project discovery.

Also a pytest plugin of the root conftest: a fixture narrower than the session takes the
temporary roots it made with it (pytest_fixture_setup)."""

from __future__ import annotations

import getpass
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast
from unittest.mock import patch

import pytest

from devtools.evaluation.golden_workspaces import remove_tree

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
    )

PROJECT_ROOT: ContextVar[Path | None] = ContextVar("pytest_project_root", default=None)
# The running test, set by the root conftest: the session seeds it copies read their market data
# and Parquet in place unless it is marked private_data (it writes them, so it gets a copy).
RUNNING_TEST: dict[str, object] = {"id": None, "private": False, "factory": None}
# This process's backup root: one per process, named once at configure. A test's own workspaces'
# generations go with the passed test, and the root with a clean session (backups_made).
BACKUP_ROOT: Path | None = None


def backups_made() -> frozenset[str]:
    """The workspaces this process's backup root holds generations for."""
    try:
        return frozenset(e.name for e in os.scandir(BACKUP_ROOT)) if BACKUP_ROOT else frozenset()
    except FileNotFoundError:
        return frozenset()


def golden_runtime():
    """Supply public OS drivers without importing the private explorer into pytest setup."""
    import portalocker  # lazy: only contemporary golden counterfactuals need these drivers.
    import psutil

    from alphalattice.control.workspace_runtime.paths import resolve_confined
    from devtools.evaluation import GoldenRuntime

    return GoldenRuntime.from_os(psutil, portalocker, resolve_confined)


def unplaced_workers(numprocesses: object, environ: dict[str, str], above: Iterable[Path]) -> str:
    """The warning for a run asking more than 2 xdist workers without a heavy place, or "".

    Only inside the private workshop, found by its tools/heavy_slot.py in ``above`` (the folders
    above the checkout): the
    public tree and CI have none, and there the place variable means nothing. It never refuses,
    since a minute-long run of at most 2 workers needs no place."""
    if environ.get("WK_HEAVY_PLACE") or not numprocesses:
        return ""
    if isinstance(numprocesses, int) and numprocesses <= 2:
        return ""
    try:
        slot = next(
            (up / "tools/heavy_slot.py" for up in above if (up / "tools/heavy_slot.py").is_file()),
            None,
        )
    # an unreadable marker is unknown: neither absence nor a reason to stop
    except OSError as error:
        return (
            f"pytest asks for {numprocesses} workers without a heavy place, and whether this is "
            f"the workshop is unknown ({error}); if it is, run it through tools/heavy_slot.py run"
            ' --line <line> --what "<what>" -- <this command>'
        )
    if slot is None:
        return ""
    return (
        f"pytest asks for {numprocesses} workers without a heavy place; run it through one: "
        f'python "{slot.as_posix()}" run --line <line> --what "<what>" -- <this command>'
    )


def warn_unplaced(config: object, above: Iterable[Path], stream: TextIO) -> None:
    """pytest_configure's controller step: write the unplaced-workers warning, if any, to
    ``stream``; it never stops the run."""
    numprocesses = config.getoption("numprocesses", None)  # type: ignore[attr-defined]
    if warning := unplaced_workers(numprocesses, dict(os.environ), above):
        print(warning, file=stream, flush=True)


def configure_temporary_roots(checkout: Path, basetemp: str | None) -> None:
    """Refuse checkout temp before collection and confine the existing backup root."""
    if basetemp is not None:
        roots = {"--basetemp": Path(os.path.abspath(str(basetemp)))}
    else:
        temproot = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir())
        try:
            user = getpass.getuser() or "unknown"
        except (OSError, KeyError):
            user = "unknown"
        roots = {
            "pytest temporary root": temproot,
            "pytest user temporary root": temproot / f"pytest-of-{user}",
            "pytest fallback temporary root": temproot / "pytest-of-unknown",
        }
    for source, path in roots.items():
        if (resolved := path.resolve()).is_relative_to(checkout):
            raise ValueError(
                f"basetemp_inside_checkout: {source} resolves to {resolved}, inside checkout "
                f"{checkout}. Refused before collection: checkout artifacts and hooks would "
                "contaminate test results. Use --basetemp with an absolute directory outside "
                "the checkout, or move PYTEST_DEBUG_TEMPROOT / TMPDIR / TEMP / TMP outside it."
            )
    backup_parent = Path(tempfile.gettempdir()).resolve()
    if backup_parent.is_relative_to(checkout):
        backup_parent = (
            Path(basetemp).resolve().parent if basetemp is not None else temproot.resolve()
        )
    # Printed commands are quoted for the shell a run started in, and a test replays one by POSIX
    # rules: under PowerShell quoting that loses a path's backslashes, and the file lands at the
    # drive root. The run's own shell must not decide that; a test of another shell sets its own.
    os.environ.setdefault("ALPHALATTICE_SHELL", "posix")
    global BACKUP_ROOT
    BACKUP_ROOT = Path(tempfile.mkdtemp(prefix="alphalattice-test-backups-", dir=backup_parent))
    os.environ["ALPHALATTICE_BACKUP_ROOT"] = str(BACKUP_ROOT)


@contextmanager
def project_scope(
    root: Path, temporary_root: Path | None = None, *, host_reads: bool = True
) -> Iterator[None]:
    """Confine discovery; a guarded unit needs the context, never a Host read adapter."""
    allowed = (temporary_root or PROJECT_ROOT.get() or root).resolve()
    assert root.resolve().is_relative_to(allowed), "project_outside_test_root"
    token = PROJECT_ROOT.set(allowed)
    try:
        if not host_reads:
            yield
            return
        from alphalattice.control.product_host.composition import first_use_road

        project, read = (
            first_use_road.session_project,
            first_use_road.read_research_workspace_manifest,
        )

        def scoped_project(start: Path, host: str | None) -> Path:
            if start == Path.cwd() and not start.resolve().is_relative_to(
                PROJECT_ROOT.get() or allowed
            ):
                start = root
            found = Path(project(start, host))
            assert found.resolve().is_relative_to(PROJECT_ROOT.get() or allowed), (
                f"project_outside_test_root: {found}"
            )
            return found

        def scoped_read(workspace: Path) -> ResearchWorkspaceManifest:
            assert workspace.resolve().is_relative_to(PROJECT_ROOT.get() or allowed), (
                f"project_outside_test_root: {workspace}"
            )
            return cast("ResearchWorkspaceManifest", read(workspace))

        with (
            patch.object(first_use_road, "session_project", scoped_project),
            patch.object(first_use_road, "read_research_workspace_manifest", scoped_read),
        ):
            yield
    finally:
        PROJECT_ROOT.reset(token)


# A fixture narrower than the session takes the temporary roots it made with it. A module's private
# workspace copy lived to the session's end: at a full run's peak, 31 of one fixture's held 3.4 GB
# (mac1, 11-07). Each root `tmp_path_factory.mktemp` makes is named after the fixture being set up,
# and goes once that fixture's own teardown is done, unless a test failed while it lived: its inputs
# are that failure's evidence. A session fixture keeps its roots to the end, and tmp_path keeps its
# own removal (tests/conftest.py), which waits for the test's verdict.
_ROOTS: list[list[Path] | None] = []
"""The roots made for each fixture being set up, innermost last; None for one that keeps them."""
_TORN: list[tuple[list[Path], Any, int]] = []
"""Roots whose fixture's teardown ran, held for its report: one that raised keeps them."""


def _record(factory: Any) -> None:
    if getattr(factory, "_fixture_roots_recorded", False):
        return
    made = factory.mktemp

    def mktemp(basename: str, numbered: bool = True) -> Path:
        path: Path = made(basename, numbered)
        if _ROOTS and _ROOTS[-1] is not None:
            _ROOTS[-1].append(path)
        return path

    factory.mktemp, factory._fixture_roots_recorded = mktemp, True


def _remove(roots: list[Path], request: Any, failed: int) -> None:
    factory = request.config._tmp_path_factory
    if request.session.testsfailed != failed or factory._retention_policy != "failed":
        return
    for root in roots:
        if os.path.lexists(root) and (left := remove_tree(root, factory.getbasetemp())):
            print(
                f"test_tmp.fixture_scratch_left: {root}: {'; '.join(left)[:500]}", file=sys.stderr
            )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """A teardown is reported once all of it ran: a fixture whose own teardown raised is a failure
    and its roots stay with the rest of its evidence."""
    if report.when != "teardown" or not _TORN:
        return
    torn, _TORN[:] = list(_TORN), []
    if not report.failed:
        for roots, request, failed in torn:
            _remove(roots, request, failed)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """A fixture finalized outside a test's teardown goes only if nothing failed while it lived."""
    torn, _TORN[:] = list(_TORN), []
    for roots, request, failed in torn:
        _remove(roots, request, failed)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Iterator[Any]:
    roots: list[Path] | None = None
    if fixturedef.scope != "session" and fixturedef.argname != "tmp_path":
        _record(request.config._tmp_path_factory)
        roots, failed = [], request.session.testsfailed
        request.addfinalizer(
            lambda: _TORN.append((roots, request, failed))
        )  # after its own teardown
    _ROOTS.append(roots)
    try:
        return (yield)
    finally:
        _ROOTS.pop()
