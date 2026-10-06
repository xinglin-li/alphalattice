"""Workspace creation, validation and run-directory resolution.

The Workspace is the root a service is given, not a factory for its own
services. It once carried `writer()` and `catalog()` convenience methods that
imported their own consumers, which made the five workspace modules one
dependency cycle for no caller: nothing in this product used either. A service
is constructed by whoever needs it, from the Workspace it operates on.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from alphalattice.control.workspace_runtime.lock import WorkspaceLock
from alphalattice.control.workspace_runtime.paths import resolve_confined, resolve_workspace_root
from alphalattice.kernel.shared_kernel.domain.errors import (
    WorkspaceCorruptionError,
    WorkspaceSecurityError,
)
from alphalattice.kernel.shared_kernel.domain.models import RunIdentity

_ROOT_DIRECTORIES = ("knowledge", "skills", "data", "experiments", ".system")
_SYSTEM_DIRECTORIES = ("staging", "recovery")
_RUN_DIRECTORIES = ("artifacts", "approvals", "reports")


class Workspace:
    """A local durable-research root whose files remain authoritative."""

    def __init__(self, root: Path) -> None:
        """Bind a workspace root and its system directory without validating the layout.

        Args:
            root: Workspace or store root used by this owner.
        """
        self.root = root
        self.system_dir = root / ".system"

    @classmethod
    def create(cls, root: str | Path) -> Workspace:
        """Create and validate the confined workspace directory layout.

        Args:
            root: Workspace or store root used by this owner.

        Returns:
            Workspace with required root and system directories.

        Raises:
            WorkspaceSecurityError: A path is not confined or is link-like.
        """
        path = Path(root).expanduser().absolute()
        path.mkdir(parents=True, exist_ok=True)
        workspace = cls(resolve_workspace_root(path))
        for name in _ROOT_DIRECTORIES:
            target = workspace.root / name
            target.mkdir(exist_ok=True)
            resolve_confined(workspace.root, name)
        for name in _SYSTEM_DIRECTORIES:
            target = workspace.system_dir / name
            target.mkdir(exist_ok=True)
            resolve_confined(workspace.root, f".system/{name}")
        return workspace

    @classmethod
    def open(cls, root: str | Path) -> Workspace:
        """Open an existing workspace after checking its required directory layout.

        Args:
            root: Workspace or store root used by this owner.

        Returns:
            Workspace rooted at the validated physical directory.

        Raises:
            WorkspaceCorruptionError: The root or a required directory is missing.
            WorkspaceSecurityError: A path is not confined or is link-like.
        """
        path = Path(root).expanduser().absolute()
        if not path.is_dir():
            raise WorkspaceCorruptionError("workspace root does not exist")
        workspace = cls(resolve_workspace_root(path))
        for name in _ROOT_DIRECTORIES:
            target = resolve_confined(workspace.root, name)
            if not target.is_dir():
                raise WorkspaceCorruptionError(f"required workspace directory is missing: {name}")
        for name in _SYSTEM_DIRECTORIES:
            target = resolve_confined(workspace.root, f".system/{name}")
            if not target.is_dir():
                raise WorkspaceCorruptionError(
                    f"required workspace system directory is missing: {name}"
                )
        return workspace

    def lock(self) -> WorkspaceLock:
        """Construct the exclusive writer lock at the confined system path.

        Returns:
            Unacquired workspace writer lock.

        Raises:
            WorkspaceSecurityError: The lock path is unsafe or hard-linked.
        """
        lock_path = resolve_confined(
            self.root,
            ".system/workspace.lock",
            reject_existing_hardlink=True,
        )
        return WorkspaceLock(lock_path)

    def run_dir(self, identity: RunIdentity, *, create: bool = False) -> Path:
        """Resolve a run directory, optionally creating its locked directory layout.

        Args:
            identity: Experiment and run identifiers selecting the run directory.
            create: Whether to create the run layout while holding the workspace lock.

        Returns:
            Existing confined run directory.

        Raises:
            WorkspaceCorruptionError: The selected run directory does not exist.
            WorkspaceSecurityError: The selected path violates confinement.
        """
        relative = f"experiments/{identity.experiment_id}/runs/{identity.run_id}"
        if create:
            with self.lock():
                target = resolve_confined(self.root, relative)
                target.mkdir(parents=True, exist_ok=True)
                target = resolve_confined(self.root, relative)
                for name in _RUN_DIRECTORIES:
                    (target / name).mkdir(exist_ok=True)
                    resolve_confined(self.root, f"{relative}/{name}")
        else:
            target = resolve_confined(self.root, relative)
        if not target.is_dir():
            raise WorkspaceCorruptionError(f"run directory does not exist: {identity.run_id}")
        return target

    def iter_run_dirs(self) -> Iterator[tuple[UUID, UUID, Path]]:
        """Iterate confined experiment/run directories in sorted directory order.

        Yields:
            Experiment UUID, run UUID and physical run path.

        Raises:
            WorkspaceSecurityError: A directory name is not a UUID or its path is unsafe.
        """
        experiments = self.root / "experiments"
        for experiment_path in sorted(experiments.iterdir()):
            if not experiment_path.is_dir():
                continue
            try:
                experiment_id = UUID(experiment_path.name)
            except ValueError:
                raise WorkspaceSecurityError(
                    f"invalid experiment directory: {experiment_path.name}"
                ) from None
            experiment_path = resolve_confined(self.root, f"experiments/{experiment_id}")
            runs_path = resolve_confined(self.root, f"experiments/{experiment_id}/runs")
            if not runs_path.is_dir():
                continue
            for run_path in sorted(runs_path.iterdir()):
                if not run_path.is_dir():
                    continue
                try:
                    run_id = UUID(run_path.name)
                except ValueError:
                    raise WorkspaceSecurityError(
                        f"invalid run directory: {run_path.name}"
                    ) from None
                run_path = resolve_confined(
                    self.root,
                    f"experiments/{experiment_id}/runs/{run_id}",
                )
                yield experiment_id, run_id, run_path
