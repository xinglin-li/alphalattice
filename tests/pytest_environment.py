"""Own pytest's temporary roots and the QA launcher's confined project discovery."""

from __future__ import annotations

import getpass
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
    )

PROJECT_ROOT: ContextVar[Path | None] = ContextVar("pytest_project_root", default=None)


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
def project_scope(root: Path, temporary_root: Path | None = None) -> Iterator[None]:
    """Confine discovery and refuse outside reads without changing the process cwd."""
    from alphalattice.control.product_host.composition import first_use_road

    allowed = (temporary_root or PROJECT_ROOT.get() or root).resolve()
    assert root.resolve().is_relative_to(allowed), "project_outside_test_root"
    project, read = first_use_road.session_project, first_use_road.read_research_workspace_manifest

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

    token = PROJECT_ROOT.set(allowed)
    try:
        with (
            patch.object(first_use_road, "session_project", scoped_project),
            patch.object(first_use_road, "read_research_workspace_manifest", scoped_read),
        ):
            yield
    finally:
        PROJECT_ROOT.reset(token)
