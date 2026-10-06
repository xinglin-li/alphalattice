"""Resolve product resources in an editable checkout or an installed distribution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def resolve_playpen_root(source_path: Path) -> Path:
    """Return the checkout root identified by both project and architecture markers."""
    resolved = source_path.resolve()
    candidates = (resolved, *resolved.parents) if resolved.is_dir() else resolved.parents
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "config" / "package-architecture.json"
        ).is_file():
            return candidate
    installed = Path(str(files("alphalattice"))) / "_runtime"
    if (installed / "config" / "identity-roles.json").is_file():
        return installed
    raise RuntimeError("shared_kernel.playpen_root_unavailable")


def source_root(root: Path) -> Path:
    """Return the import root without changing canonical source component names.

    Args:
        root: The checkout or packaged resource root.

    Returns:
        The directory containing the alphalattice package.
    """
    if root.name == "_runtime" and root.parent.name == "alphalattice":
        return root.parent.parent
    return root / "src"


def resource_path(root: Path, relative: str) -> Path:
    """Resolve a canonical resource path on either runtime leg.

    Args:
        root: The checkout or packaged resource root.
        relative: The repository-relative resource name.

    Returns:
        The resource's physical file.
    """
    path = Path(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("shared_kernel.resource_path_invalid")
    return source_root(root).joinpath(*path.parts[1:]) if path.parts[0] == "src" else root / path


def tracked_path(root: Path, path: Path) -> str:
    """Keep identity input names independent of the installation layout.

    Args:
        root: The checkout or packaged resource root.
        path: A source or resource file.

    Returns:
        Its canonical repository-relative name.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    if resolved.is_relative_to(source_root(root)):
        return "src/" + resolved.relative_to(source_root(root)).as_posix()
    raise ValueError("shared_kernel.source_resource_outside_runtime")


def command_prefix() -> tuple[str, ...]:
    """Return the tool-installed command, available in every host's fresh shell."""
    return ("alphalattice",)
