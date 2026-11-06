"""Resolve product resources in an editable checkout or an installed distribution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import suppress
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

REFUSAL_WORDS_RESOURCE: Final = "src/alphalattice/interface/local_application/refusal_words.json"


@cache
def runtime_json(relative: str) -> dict[str, Any]:
    """Read one shipped JSON resource once per process, on either runtime leg."""
    root = resolve_playpen_root(Path(__file__))
    payload: dict[str, Any] = json.loads(resource_path(root, relative).read_text(encoding="utf-8"))
    return payload


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


def command_prefix(*, portable: bool = False) -> tuple[str, ...]:
    """Return the tool-installed command, available in every host's fresh shell."""
    if portable or ("UV_RUN_RECURSION_DEPTH" not in os.environ and shutil.which("alphalattice")):
        return ("alphalattice",)
    uv = shutil.which("uv")
    folder = None
    if uv:
        # Catch outside @cache so a failed probe cannot become a retained value.
        with suppress(OSError, subprocess.SubprocessError, UnicodeError):
            folder = _tool_bin(uv)
    command = shutil.which("alphalattice", path=folder) if folder else None
    if command:
        return (str(Path(command).absolute()),)
    return ("alphalattice",)


@cache
def _tool_bin(uv: str) -> str:
    """Cache successful directory discovery; failed probes remain retriable."""
    found = subprocess.run(
        [uv, "tool", "dir", "--bin"],
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    ).stdout.strip()
    if not found:
        raise OSError
    return found
