"""Own pytest's temporary roots and the QA launcher's confined project discovery."""

from __future__ import annotations

import getpass
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast
from unittest.mock import patch

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
    )

PROJECT_ROOT: ContextVar[Path | None] = ContextVar("pytest_project_root", default=None)
# The running test, set by the root conftest: the session seeds it copies read their market data
# and Parquet in place unless it is marked private_data (it writes them, so it gets a copy).
RUNNING_TEST: dict[str, str | bool | None] = {"id": None, "private": False}


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
    os.environ["ALPHALATTICE_BACKUP_ROOT"] = tempfile.mkdtemp(
        prefix="alphalattice-test-backups-", dir=backup_parent
    )


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
